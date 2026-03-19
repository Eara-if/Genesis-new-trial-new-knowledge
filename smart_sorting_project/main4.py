import numpy as np
from smart_gripper import SmartParallelGripper
from robot_env import setup_environment
from motion_planner import LocalPlanner

def main():
    scene, franka, table, legs, cube_list = setup_environment()
    planner = LocalPlanner(scene, franka, [table] + legs)
    gripper = SmartParallelGripper(franka, planner.fingers_dof)
    
    # Franka 夹爪指尖到手腕中心的物理距离偏移
    GRIPPER_TCP_OFFSET = 0.104 
    # 安全高度余量（防止撞倒物块）
    SAFETY_MARGIN = 0.01 
    
        # ... 在 main.py 的循环中 ...

    for item in cube_list:
        entity = item["entity"]
        c_aabb = entity.get_AABB()
        z_max = c_aabb[1][2].cpu().numpy() # 目标物块顶部的绝对高度
        curr_pos = entity.get_pos().cpu().numpy()
    
        # 物理补偿
        GRIPPER_OFFSET = 0.104
    
        # 【修复点】计算两个关键点
        # 1. 悬停点：就在目标物块的正上方，但高度要高出 10cm 以上，确保能跨过路上的障碍
        hover_pos = np.array([curr_pos[0], curr_pos[1], z_max + GRIPPER_OFFSET + 0.10])
    
        # 2. 抓取点：最终接触物块的位置
        grasp_pos = np.array([curr_pos[0], curr_pos[1], z_max + GRIPPER_OFFSET - 0.005])

        print(">>> 动作 A：高空横移，避开地面障碍物")
        # 这一步非常关键：即便在桌底，也要先去 hover_pos
        # 如果路途中有其他方块，因为 hover_pos 足够高，机械臂会从它们头顶飞过去
        planner.move_to(hover_pos, avoid_obstacles=True, speed=0.04)

        print(">>> 动作 B：垂直降落，开始夹取")
        # 此时位置已经对准了，再垂直降落，就不会在地面“贴地推行”撞到前面的方块
        planner.move_to(grasp_pos, avoid_obstacles=False, speed=0.01)

        # 3. 抓取
        f = -max(gripper.compute_required_force(item['mass'], item['mu']) * 2.0, 30.0)
        for _ in range(80):
            q = franka.inverse_kinematics(link=planner.ee_link, pos=grasp_pos, quat=np.array([0,1,0,0]))
            if q is not None: franka.control_dofs_position(q[:7], planner.motors_dof)
            franka.control_dofs_force(np.array([f, f]), planner.fingers_dof)
            scene.step()

        # 4. 垂直抬升并放置
        planner.move_to(grasp_pos + [0, 0, 0.15], force=f, avoid_obstacles=False)
        planner.move_to(item["target_pos"] + [0, 0, 0.2], force=f, avoid_obstacles=True)
        
        # 释放
        franka.control_dofs_position(np.array([0.04, 0.04]), planner.fingers_dof)
        for _ in range(60): scene.step()

if __name__ == "__main__":
    main()