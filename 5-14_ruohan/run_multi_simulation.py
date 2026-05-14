import os
import sys
import random
import gc
import argparse

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

if 'QT_QPA_PLATFORM' in os.environ:
    del os.environ['QT_QPA_PLATFORM']

import genesis as gs
import numpy as np
import cv2
import torch

cv2.setNumThreads(0)
torch.set_num_threads(1)

from vision_processor import VisionProcessor
from waypoint_multi_task import WaypointMultiPickTask
from sensor_manager import SensorManager
from LeWM_writer import DatasetWriter
import assets_manager as am

def enforce_version():
    required_version = "0.3.13"
    if gs.__version__ != required_version:
        print(f"⚠️ [系统拦截] 当前 Genesis 版本为 {gs.__version__}！请回滚至 {required_version}。")
        sys.exit(1)

def main():
    # ==========================================
    # 解析命令行参数，控制无头模式
    # ==========================================
    parser = argparse.ArgumentParser(description="Genesis 批量数据生成流水线")
    parser.add_argument('--vis', action='store_true', help="开启可视化模式（默认关闭，追求极致速度）")
    parser.add_argument('--episodes', type=int, default=100, help="要生成的总局数 (默认 100)")
    args = parser.parse_args()

    ENABLE_VIS = args.vis
    TOTAL_EPISODES = args.episodes

    if ENABLE_VIS:
        print("👁️ 可视化模式已开启 (速度较慢，适用于调试)")
        cv2.namedWindow("Vision Processing", cv2.WINDOW_AUTOSIZE)
    else:
        print("🚀 无头模式已开启 (最高速度生成数据，无画面输出)")

    enforce_version()
    gs.init(backend=gs.cpu)

    writer = DatasetWriter(repo_id="genesis_multi_task", root_dir="outputs/lewm_dataset", fps=50)

    for ep in range(TOTAL_EPISODES):
        print(f"\n{'='*60}")
        print(f"🎬 开始生成第 {ep + 1}/{TOTAL_EPISODES} 局专家数据...")
        print(f"{'='*60}")

        # 1. 初始化场景与实体
        scene = gs.Scene(
            show_viewer=ENABLE_VIS, 
            sim_options=gs.options.SimOptions(dt=0.01, substeps=20),
            vis_options=gs.options.VisOptions(segmentation_level='entity') 
        )

        scene.add_entity(gs.morphs.Plane())
        robot = scene.add_entity(gs.morphs.MJCF(file='xml/universal_robots_ur5e/ur5e.xml'))

        is_stacked_task = random.choice([True, False])
        num_boxes = random.randint(2, 4)
        obj_list = am.spawn_multiple_boxes(scene, ['box_1.jpg', 'box_2.jpg'], count=num_boxes, is_stacked=is_stacked_task)

        prompts = [
            f"Clean up the {num_boxes} boxes on the table.",
            f"Clear all objects into the bin, they are {'stacked' if is_stacked_task else 'scattered'}.",
            f"Pick and place the {'tower of' if is_stacked_task else ''} target boxes."
        ]
        current_prompt = random.choice(prompts)

        sensor_manager = SensorManager(scene=scene, robot=robot, obj_list=obj_list, image_size=(640, 480))
        
        # 2. 编译场景（必须在操控机器人前调用）
        scene.build()

        # =========================================================
        # 🛡️ 机械臂安全复位 (Safe Reset) 机制
        # 必须在 scene.build() 之后执行！
        # =========================================================
        # 定义 UR5e 标准安全观察姿态 (Home Position)
        safe_home_qpos = np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])

        # 强制瞬间改变关节的真实物理位置 (绕过控制器，直接瞬移)
        robot.set_dofs_position(safe_home_qpos, np.arange(6))

        # 强制清零所有关节的物理速度和加速度 (刹车)
        robot.set_dofs_velocity(np.zeros(6), np.arange(6))

        # 把控制器的目标点也同步到当前位置，防止 PD 控制器一通电就乱拉
        robot.control_dofs_position(safe_home_qpos, np.arange(6))

        # 让物理引擎在无动作指令的情况下，空跑几步“消化”一下重置状态
        for _ in range(10):
            scene.step()
        # =========================================================

        # 3. 启动视觉处理与任务控制器
        vp = VisionProcessor()
        task = WaypointMultiPickTask(scene, robot, obj_list)
        # task.start() 内部会读取刚才稳定下来的安全复位坐标作为初始 goal
        task.start()

        writer.start_episode(metadata={"task": current_prompt})

        step_counter = 0
        print(f"🚀 [运行中] 当前任务: {current_prompt}")

        try:
            while True:
                is_all_completed = task.step()
                scene.step()

                if step_counter % 2 == 0: 
                    sensors = sensor_manager.capture_all()
                    expert_data = task.get_expert_data()
                    writer.write_step(step_counter, {"sensors": sensors, "expert_data": expert_data})

                    # 只有在开启可视化时，才进行画面渲染和 OpenCV 操作
                    top_payload = sensors.get("top_cam")
                    if top_payload:
                        rgb_raw = top_payload["rgb"]
                        depth_raw = top_payload["depth"]
                        
                        if task.phase == 'searching':
                            bev_view, detected_data = vp.get_bev_and_data(rgb_raw, depth_raw)
                            task.update_targets(detected_data)
                            
                            if ENABLE_VIS and bev_view is not None:
                                cv2.imshow("Vision Processing", bev_view)
                        else:
                            if ENABLE_VIS:
                                bgr_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
                                cv2.imshow("Vision Processing", bgr_raw)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("⚠️ 检测到手动按下 'q'，跳过当前局...")
                    break

                if is_all_completed:
                    print(f"🎯 第 {ep + 1} 局任务已成功完成！")
                    break

                step_counter += 1

        except KeyboardInterrupt:
            print("\n⚠️ 检测到终端 Ctrl+C 中断，正在安全保存退出...")
            writer.end_episode()
            writer.finish_dataset()
            if ENABLE_VIS:
                cv2.destroyAllWindows()
            sys.exit(0)

        writer.end_episode()
        
        am.clear_temp_urdfs() 
        del scene             
        del sensor_manager
        del task
        del vp
        gc.collect()          
        torch.cuda.empty_cache() if torch.cuda.is_available() else None 

    writer.finish_dataset()
    if ENABLE_VIS:
        cv2.destroyAllWindows()
    print(f"🎉 自动化采集完毕！共计生成了 {TOTAL_EPISODES} 局专家轨迹。")

if __name__ == "__main__":
    main()