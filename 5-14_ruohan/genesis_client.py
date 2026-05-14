# genesis_client.py (运行在 Genesis conda 环境下)
import cv2
import numpy as np
import requests
import time

# 导入你项目的相关依赖
import genesis as gs
from sensor_manager import SensorManager
import assets_manager as am

BRAIN_URL = "http://127.0.0.1:8000/predict_action"

def main():
    print("[*] 正在初始化 Genesis 物理仿真环境...")
    gs.init(backend=gs.cpu)
    
    scene = gs.Scene(
        show_viewer=True, 
        sim_options=gs.options.SimOptions(dt=0.01, substeps=20),
        vis_options=gs.options.VisOptions(segmentation_level='entity') 
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file='xml/universal_robots_ur5e/ur5e.xml'))
    
    obj_list = am.spawn_multiple_boxes(scene, ['box_1.jpg', 'box_2.jpg'], count=3, is_stacked=False)
    sensor_manager = SensorManager(scene=scene, robot=robot, obj_list=obj_list, image_size=(640, 480))
    scene.build()

    ee_link = robot.links[7]
    top_quat = np.array([0, 1, 0, 0], dtype=np.float64) 
    
    # ---------------------------------------------------------
    # 🛡️ 机械臂安全复位 (Safe Reset) 机制
    # 注意：这段代码必须在 while True 循环外部！否则机械臂每帧都会归零
    # ---------------------------------------------------------
    print("[*] 执行机械臂安全复位...")
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
        
    current_ee_pos = ee_link.get_pos().detach().cpu().numpy().astype(np.float64)
    # ---------------------------------------------------------

    print("\n🚀 开始在线闭环测试 (正在使用训练好的策略网络)...")
    
    step_counter = 0
    while True:
        scene.step()
        
        # 每 10 个物理步长，向大脑请求一次动作
        if step_counter % 10 == 0:
            sensors = sensor_manager.capture_all()
            top_cam_rgb = sensors.get("top_cam", {}).get("rgb")
            
            if top_cam_rgb is not None:
                # 1. 压缩图像
                bgr_img = cv2.cvtColor(top_cam_rgb, cv2.COLOR_RGB2BGR)
                _, img_encoded = cv2.imencode('.jpg', bgr_img)
                
                # 2. 发送给 LeWM 大脑
                try:
                    t_start = time.time()
                    response = requests.post(
                        BRAIN_URL, 
                        files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")},
                        timeout=1.0
                    )
                    
                    if response.status_code == 200:
                        # 3. 接收策略网络(小脑)返回的 25 维动作
                        action_data = response.json().get("action")
                        action_vector = np.array(action_data, dtype=np.float32)
                        
                        # 4. 提取位移并执行动作
                        delta_pos = action_vector[:3]
                        
                        # 限制单步最大移动，防止策略输出异常导致机器人飞掉
                        max_delta = 0.02
                        dist = np.linalg.norm(delta_pos)
                        if dist > max_delta:
                            delta_pos = (delta_pos / dist) * max_delta

                        current_ee_pos += delta_pos
                        q_target = robot.inverse_kinematics(link=ee_link, pos=current_ee_pos, quat=top_quat)
                        
                        if q_target is not None:
                            robot.control_dofs_position(q_target[:6], np.arange(6))
                            
                    else:
                        print(f"⚠️ 服务器返回异常状态码: {response.status_code}")
                    
                except requests.exceptions.ConnectionError:
                    print("⚠️ 无法连接到大脑服务器，请确认 lewm_server.py 正在运行！")
                except Exception as e:
                    print(f"⚠️ 通信发生错误: {e}")

            # 显示实时画面
            if top_cam_rgb is not None:
                cv2.imshow("Robot View", bgr_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("🛑 手动终止仿真。")
                    break

        step_counter += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()