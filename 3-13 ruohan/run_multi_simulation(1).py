import genesis as gs
import numpy as np
import cv2

from vision_processor import VisionProcessor
from waypoint_multi_task import WaypointMultiPickTask
from sensor_manager import SensorManager
from dataset_writer import DatasetWriter
import assets_manager as am

# ==========================================
# 场景配置开关：
# True  -> 使用堆栈叠放模式
# False -> 使用随机散落模式
# ==========================================
USE_STACKED_SCENE = False

def main():
    gs.init(backend=gs.vulkan)

    # 🛑 核心修复点 1：全局锁定渲染管线的分割层级为 'entity'
    scene = gs.Scene(
        show_viewer=True,
        sim_options=gs.options.SimOptions(dt=0.01, substeps=20),
        vis_options=gs.options.VisOptions(segmentation_level='entity') 
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file='xml/universal_robots_ur5e/ur5e.xml'))

    obj_list = am.spawn_multiple_boxes(
        scene,
        ['box_1.jpg', 'box_2.jpg'],
        count=3,
        is_stacked=USE_STACKED_SCENE
    )

    sensor_manager = SensorManager(
        scene=scene,
        robot=robot,
        obj_list=obj_list,
        image_size=(640, 480)
    )

    scene.build()

    print("\n" + "="*60)
    print("🔍 [诊断模式] 正在提取底层渲染器的原始分割数据...")
    scene.step() 
    
    debug_cam = sensor_manager.cameras.get("top_cam")
    if debug_cam:
        try:
            res = debug_cam.render(rgb=True, depth=True, segmentation=True)
            debug_raw_seg = res[2]
            unique_ids = np.unique(debug_raw_seg)
            print(f"📊 当前场景中实际渲染出的唯一 ID 集合: {unique_ids}")
            print("💡 预期结果：现在 ID 应该紧凑且极小（如 [1, 2, 3, 4, 5]），不再有十几的碎片数字！")
        except Exception as e:
            print(f"❌ 诊断渲染失败: {e}")
    print("="*60 + "\n")

    vp = VisionProcessor()
    task = WaypointMultiPickTask(scene, robot, obj_list)
    task.start()

    writer = DatasetWriter(
        root_dir="outputs/dataset",
        prefix="multi_modal",
        save_every_n_steps=5
    )

    writer.start_episode(metadata={
        "use_stacked_scene": USE_STACKED_SCENE,
        "num_objects": len(obj_list),
        "cameras": list(sensor_manager.cameras.keys()),
    })

    step_counter = 0

    while True:
        task.step()
        scene.step()

        if step_counter % 5 == 0:
            sensors = sensor_manager.capture_all()
            top_payload = sensors.get("top_cam")
            if top_payload is None:
                continue

            rgb_raw = top_payload["rgb"]
            depth_raw = top_payload["depth"]
            bgr_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)

            if task.phase == 'searching':
                bev_view, detected_data = vp.get_bev_and_data(rgb_raw, depth_raw)
                task.update_targets(detected_data)

                if bev_view is not None:
                    cv2.putText(
                        bev_view,
                        "VISION: ACTIVE (UPDATING)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    cv2.imshow("Vision Processing", bev_view)
            else:
                cv2.putText(
                    bgr_raw,
                    f"VISION: FROZEN | PHASE: {task.phase.upper()}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                cv2.imshow("Vision Processing", bgr_raw)

            front_payload = sensors.get("front_cam")
            if front_payload is not None:
                cv2.imshow(
                    "Front RGB",
                    cv2.cvtColor(front_payload["rgb"], cv2.COLOR_RGB2BGR)
                )
                cv2.imshow("Front Segmentation", front_payload["seg_color"])

            wrist_payload = sensors.get("wrist_cam")
            if wrist_payload is not None:
                cv2.imshow(
                    "Wrist RGB",
                    cv2.cvtColor(wrist_payload["rgb"], cv2.COLOR_RGB2BGR)
                )
                cv2.imshow("Wrist Segmentation", wrist_payload["seg_color"])

            tactile = task.gripper.get_tactile_data()
            if tactile is not None:
                pressure_vis = (np.clip(tactile["pressure_map"], 0.0, 1.0) * 255).astype(np.uint8)
                pressure_vis = cv2.applyColorMap(
                    cv2.resize(pressure_vis, (256, 256), interpolation=cv2.INTER_NEAREST),
                    cv2.COLORMAP_JET
                )
                cv2.imshow("Tactile Pressure", pressure_vis)

            writer.write_step(step_counter, {
                "sensors": sensors,
                "tactile": tactile,
                "meta": {
                    "phase": task.phase,
                    "num_detected_targets": len(task.target_list),
                    "object_states": sensor_manager.get_object_states(),
                },
            })

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        step_counter += 1

    if USE_STACKED_SCENE:
        am.clear_temp_urdfs()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()