import numpy as np
from smart_gripper import SmartParallelGripper
from robot_env import setup_environment
from motion_planner import LocalPlanner
from data_collector import DreamerDataCollector

def main():
    # 1. 初始化
    scene, franka, table, legs, cube_list, cam = setup_environment()
    planner = LocalPlanner(scene, franka, [table] + legs)
    gripper = SmartParallelGripper(franka, planner.fingers_dof)
    collector = DreamerDataCollector()
    
    # --- 【关键修复 1：物理参数加固】 ---
    # 强制提升手指的 KP（刚度），确保它能对抗渲染带来的延迟和物理波动
    # 这里的 5000 是经验值，如果还张不开可以加到 8000
    franka.set_dofs_kp(np.array([5000, 5000]), planner.fingers_dof)
    franka.set_dofs_kv(np.array([100, 100]), planner.fingers_dof)

    # 数据采集回调
    def record_step_callback(action):
        # 如果 action 是 Tensor，先转 CPU Numpy
        if hasattr(action, 'cpu'): action = action.cpu().numpy()
        
        rgb = cam.render()[0]
        state = franka.get_dofs_position().cpu().numpy()
        collector.add_step(raw_img=rgb, action=action, state=state, reward=0, done=False, is_first=False)

    GRIPPER_OFFSET = 0.104

    for i, item in enumerate(cube_list):
        entity = item["entity"]
        c_aabb = entity.get_AABB()
        z_max = c_aabb[1][2].cpu().numpy()
        curr_pos = entity.get_pos().cpu().numpy()
    
        hover_pos = np.array([curr_pos[0], curr_pos[1], z_max + GRIPPER_OFFSET + 0.10])
        grasp_pos = np.array([curr_pos[0], curr_pos[1], z_max + GRIPPER_OFFSET - 0.005])

        # 记录 Episode 第一帧
        collector.add_step(cam.render()[0], np.zeros(7), franka.get_dofs_position().cpu().numpy(), 0, False, True)

        print(f">>> 正在处理第 {i} 个物块: 动作 A - 避障平移")
        planner.move_to(hover_pos, avoid_obstacles=True, speed=0.04, callback=record_step_callback)

        print(">>> 动作 B：垂直降落")
        planner.move_to(grasp_pos, avoid_obstacles=False, speed=0.01, callback=record_step_callback)

        # 3. 抓取 (下潜对准与闭合)
        print(">>> 动作 C：原地对准并闭合夹取")
        f_grip = -max(gripper.compute_required_force(item['mass'], item['mu']) * 2.0, 30.0)
        
        # 强制位置对准并保持手指张开
        for _ in range(40):
            q = franka.inverse_kinematics(link=planner.ee_link, pos=grasp_pos, quat=np.array([0,1,0,0]))
            if q is not None:
                action = q[:7]
                franka.control_dofs_position(action, planner.motors_dof)
                # --- 【关键修复 2：双保险张开】 ---
                franka.control_dofs_position(np.array([0.045, 0.045]), planner.fingers_dof)
            scene.step()
            record_step_callback(action=action)

        # 发力抓紧
        for _ in range(40):
            franka.control_dofs_force(np.array([f_grip, f_grip]), planner.fingers_dof)
            scene.step()
            record_step_callback(action=action)

        # 4. 抬升与放置
        planner.move_to(grasp_pos + [0, 0, 0.15], force=f_grip, avoid_obstacles=False, callback=record_step_callback)
        planner.move_to(item["target_pos"] + [0, 0, 0.2], force=f_grip, avoid_obstacles=True, callback=record_step_callback)
        
        # 释放
        franka.control_dofs_position(np.array([0.04, 0.04]), planner.fingers_dof)
        for _ in range(60): 
            scene.step()
            record_step_callback(action=franka.get_dofs_position().cpu().numpy()[:7])

        collector.save_episode(episode_idx=i)

if __name__ == "__main__":
    main()