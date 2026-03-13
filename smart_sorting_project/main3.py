import numpy as np
import genesis as gs
from smart_gripper import SmartParallelGripper
from assets_manager import spawn_random_boxes

########################## 1. 初始化 ##########################
gs.init(backend=gs.gpu, precision='32')
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.5, 2.0, 1.5), 
        camera_lookat=(0.4, 0, 0.2)
    ),
    show_viewer=True
)

########################## 2. 实体加载 ##########################
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
ee_link = franka.get_link('hand')

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# 障碍物
obstacles = []
obstacles.append(scene.add_entity(
    gs.morphs.Box(size=(0.035, 0.26, 0.25), pos=(0.35, 0.25, 0.125)),  
    surface=gs.surfaces.Default(color=(1.0, 0.2, 0.2))
))
obstacles.append(scene.add_entity(
    gs.morphs.Box(size=(0.035, 0.26, 0.25), pos=(0.35, -0.25, 0.125)), 
    surface=gs.surfaces.Default(color=(1.0, 0.2, 0.2))
))
obstacles.append(scene.add_entity(
    gs.morphs.Box(size=(0.035, 0.035, 0.3), pos=(0.5, -0.35, 0.15)), 
    surface=gs.surfaces.Default(color=(1.0, 0.2, 0.2))
))
obstacles.append(scene.add_entity(
    gs.morphs.Box(size=(0.035, 0.035, 0.3), pos=(0.5, 0.35, 0.15)), 
    surface=gs.surfaces.Default(color=(1.0, 0.2, 0.2))
))

cube_list, _ = spawn_random_boxes(scene, count=4)
scene.build()

########################## 3. 避障控制算法 ##########################

def move_intelligent_final(target_pos, finger_pos=0.045, force=0.0, speed=0.03, min_height=0.4, avoid_obstacles=True):
    links_to_check = [franka.get_link(name) for name in ['link5', 'link6', 'link7', 'hand']]

    for step in range(800):
        curr_ee_pos = ee_link.get_pos().cpu().numpy()
        dist_to_goal = np.linalg.norm(target_pos - curr_ee_pos)
        
        if dist_to_goal < 0.008: break # 提高精度到 8mm
        
        repulsive_vec = np.zeros(3)
        # 只有在开启避障模式时，才计算红板子的斥力
        if avoid_obstacles:
            for link in links_to_check:
                link_pos = link.get_pos().cpu().numpy()
                for obs in obstacles:
                    obs_pos = obs.get_pos().cpu().numpy()
                    dist = np.linalg.norm(link_pos - obs_pos)
                    if dist < 0.25: 
                        w = (0.25 - dist) / 0.25
                        repulsive_vec += np.array([0, 0, 0.2]) * w
                        side_push = (link_pos - obs_pos)
                        side_push[2] = 0 
                        repulsive_vec += (side_push / (np.linalg.norm(side_push) + 1e-6)) * 0.15 * w

        modified_target = target_pos.copy()
        # 只有在避障模式下，才强制高空巡航
        if avoid_obstacles and dist_to_goal > 0.15: 
            modified_target[2] = max(target_pos[2], min_height) 

        diff = modified_target - curr_ee_pos
        attraction = diff / (np.linalg.norm(diff) + 1e-6)
        
        # 混合方向
        move_dir = attraction * speed + repulsive_vec
        
        # 限制步幅
        if np.linalg.norm(move_dir) > speed:
            move_dir = move_dir / np.linalg.norm(move_dir) * speed
            
        next_pos = curr_ee_pos + move_dir
        
        q_goal = franka.inverse_kinematics(link=ee_link, pos=next_pos, quat=np.array([0, 1, 0, 0]))
        if q_goal is not None:
            franka.control_dofs_position(q_goal[:7], motors_dof)
            if force == 0:
                franka.control_dofs_position(np.array([finger_pos, finger_pos]), fingers_dof)
            else:
                franka.control_dofs_force(np.array([force, force]), fingers_dof)
        
        scene.step()

########################## 4. 优化后的分拣流程 ##########################

franka.set_dofs_kp(np.array([5000]*7 + [100, 100]))
franka.set_dofs_kv(np.array([500]*7 + [10, 10]))
smart_gripper = SmartParallelGripper(franka, fingers_dof)

for item in cube_list:
    # 每一轮抓取前重新刷新位置，防止物体被碰歪
    c_pos = item["entity"].get_pos().cpu().numpy()
    t_pos = item["target_pos"]

    print(f">>> 1. 智能避障移动：前往方块上方")
    # 此时 avoid_obstacles=True，机械臂会优雅地绕过红板子
    move_intelligent_final(c_pos + [0, 0, 0.20], finger_pos=0.045, speed=0.04, avoid_obstacles=True)

    print(f">>> 2. 稳健下压：关闭避障逻辑，死磕精度")
    # 此时 avoid_obstacles=False，机械臂像钻头一样垂直降落，不会受红板子干扰而抖动
    grasp_height = item['dims'][2]/2 + 0.095 
    move_intelligent_final(np.array([c_pos[0], c_pos[1], grasp_height]), 
                           finger_pos=0.045, speed=0.01, avoid_obstacles=False)
    
    # 原地绝对静止 50 步
    for _ in range(50): scene.step()

    print(f">>> 3. 闭合抓取：锁定位置")
    f_val = -max(smart_gripper.compute_required_force(item['mass'], item['mu']) * 3.0, 35.0)
    for _ in range(120):
        # 抓取时必须持续锁定位置，防止反作用力导致手臂漂移
        q_stay = franka.inverse_kinematics(link=ee_link, pos=[c_pos[0], c_pos[1], grasp_height], quat=np.array([0,1,0,0]))
        if q_stay is not None:
            franka.control_dofs_position(q_stay[:7], motors_dof)
        franka.control_dofs_force(np.array([f_val, f_val]), fingers_dof)
        scene.step()

    print(f">>> 4. 拎高：垂直拔起")
    # 先垂直向上离开危险区，再开启避障
    move_intelligent_final(ee_link.get_pos().cpu().numpy() + [0, 0, 0.15], force=f_val, avoid_obstacles=False)
    
    print(f">>> 5. 智能运送：带着物块避障前往回收区")
    move_intelligent_final(t_pos + [0, 0, 0.25], force=f_val, avoid_obstacles=True, min_height=0.45)
    
    # 6. 放置逻辑保持不变，但建议放置时也关闭 avoid_obstacles 以求平稳
    move_intelligent_final([t_pos[0], t_pos[1], 0.12], force=f_val, speed=0.02, avoid_obstacles=False)
    
    # 释放...
    franka.control_dofs_position(np.array([0.045, 0.045]), fingers_dof)
    for _ in range(100): scene.step()
    
    # 撤离...
    move_intelligent_final(ee_link.get_pos().cpu().numpy() + [0, 0, 0.2], avoid_obstacles=True)

print(">>> 分拣演示完成。")