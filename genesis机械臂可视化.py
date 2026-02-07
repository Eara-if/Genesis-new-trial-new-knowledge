import numpy as np
import genesis as gs
import random
import time

########################## 1. 初始化 ##########################
gs.init(backend=gs.gpu)

scene = gs.Scene(
    # 增加子步数 (substeps) 到 5，这是解决穿模最有效的参数
    # 它会让物理引擎在一帧内多计算几次碰撞，防止物体“穿透”
    sim_options=gs.options.SimOptions(dt=0.005, substeps=5), 
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.5, 0.0, 1.2),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
    ),
    show_viewer=True,
    rigid_options=gs.options.RigidOptions(
        batch_dofs_info=True,
        batch_links_info=True,
    ),
)

########################## 2. 实体加载 ##########################
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
ee_link = franka.get_link('hand')

cubes = {
    0: scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0, 0, -1.0)), surface=gs.surfaces.Rough(color=(1.0, 0.2, 0.2))),
    1: scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0, 0, -1.0)), surface=gs.surfaces.Rough(color=(0.2, 0.2, 1.0))),
    2: scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0, 0, -1.0)), surface=gs.surfaces.Rough(color=(0.2, 1.0, 0.2)))
}

scene.build()

########################## 3. 控制增益设置 ##########################
motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
)

########################## 4. 动作函数 ##########################
def get_ik_qpos(target_pos, target_quat):
    return franka.inverse_kinematics(
        link=ee_link,
        pos=target_pos,
        quat=target_quat,
    )

def move_path(target_pos, num_waypoints=100):
    target_quat = np.array([0, 1, 0, 0])
    qpos_goal = get_ik_qpos(target_pos, target_quat)
    
    # 保持手指当前状态
    current_fingers = franka.get_dofs_position()[-2:]
    qpos_goal[-2:] = current_fingers

    path = franka.plan_path(
        qpos_goal=qpos_goal,
        num_waypoints=num_waypoints,
    )
    
    if path is not None:
        for waypoint in path:
            franka.control_dofs_position(waypoint)
            scene.step()
    
    for _ in range(10): scene.step()

def gripper_action(width, force=None):
    # width: 目标宽度 (0.04 是全开, 0.0 是全闭)
    current_q = franka.get_dofs_position().cpu().numpy()
    target_q = current_q.copy()
    
    # Franka 的 fingers 是两个关节，每个关节移动距离是总宽度的一半
    # 如果我们要张开到 0.04m (4cm)，每个手指大概是 0.04
    target_q[-2:] = width 
    
    if force is not None:
        # 力控制
        franka.control_dofs_position(target_q[:-2], motors_dof)
        franka.control_dofs_force(np.array([force, force]), fingers_dof)
    else:
        # 位置控制
        franka.control_dofs_position(target_q)
        
    for _ in range(50): scene.step()

########################## 5. 主循环 ##########################
COLOR_MAP = {
    0: {"name": "red",   "target": np.array([0.4, -0.3, 0.2])},
    1: {"name": "blue",  "target": np.array([0.4,  0.3, 0.2])},
    2: {"name": "green", "target": np.array([0.3,  0.0, 0.2])}
}

# [调整 1] 你的实测最佳参数
GRIPPER_LENGTH = 0.11 

print(">>> 修正物理穿模版启动...")

for ep in range(50):
    # --- [修改点：在循环最开始加入这行] ---
    try: scene.sim.rigid_solver.clear_constraints() 
    except: pass
    c_id = random.choice([0, 1, 2])
    c_info = COLOR_MAP[c_id]
    
    for i in cubes: cubes[i].set_pos(np.array([0, 0, -2.0]))
    franka.set_dofs_position(np.array([0, 0, 0, -1.57, 0, 1.57, 0.78, 0.04, 0.04]))
    
    active_cube = cubes[c_id]
    rx, ry = random.uniform(0.5, 0.6), random.uniform(-0.1, 0.1)
    # 物体高度 0.02 (贴地)
    active_cube.set_pos(np.array([rx, ry, 0.02]))
    
    # 清理约束
    try: 
        # 注意：这里我们手动管理约束对象，不需要调用 clear_constraints
        # scene.sim.rigid_solver.clear_constraints() 
        pass
    except: pass
    
    for _ in range(20): scene.step()
    
    print(f"Ep {ep} | 目标: {c_info['name']}")

    # --- [修改点：在这里加一行初始化] ---
    constraint = None 
    # ----------------------------------
    
    try:
        cur_pos = active_cube.get_pos().cpu().numpy()
        
        # 1. 预备 (全开，在方块上方 20cm 左右)
        gripper_action(0.04) 
        move_path(cur_pos + [0, 0, 0.25], num_waypoints=40)
        
        # 2. 下探 (对齐官方 0.11 的偏移量)
        # cur_pos[2] 是 0.02，加上 0.11 等于 0.13，完美对齐官方示例
        move_path(cur_pos + [0, 0, 0.13], num_waypoints=40)
        
        # 3. 物理抓取 (加大力度，确保捏死)
        # 宽度 0.015 小于物块 0.04，确保有挤压；力用 -20.0
        gripper_action(0.015, force=-20.0)
        
        # [额外增加] 停顿一下，让物块被捏稳
        for _ in range(20): scene.step()
        
        # 4. 提起
        # 此时手指应该紧紧夹着物块
        move_path(cur_pos + [0, 0, 0.3], num_waypoints=40)
        
        # 5. 移动到目标上方
        move_path(c_info["target"] + [0, 0, 0.2], num_waypoints=60)
        
        # 6. 释放
        gripper_action(0.04)
        
        # 7. 离开
        move_path(c_info["target"] + [0, 0, 0.3], num_waypoints=30)
        
    except Exception as e:
        print(f"❌ Ep {ep} 出错: {e}")
        # 出错时也要确保约束被切断
        if constraint is not None:
            try: constraint.destroy()
            except: pass
        time.sleep(1)

    for _ in range(30): scene.step()