import numpy as np
import genesis as gs
import random

########################## 1. 初始化 (物理加强) ##########################
gs.init(backend=gs.gpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.005, substeps=10), # 增加子步提高约束稳定性
    viewer_options=gs.options.ViewerOptions(
        camera_pos    = (2.0, 1.5, 1.5),
        camera_lookat = (0.5, 0.0, 0.5),
        camera_fov    = 40,
    ),
    
    show_viewer=True,
)

########################## 2. 实体加载 (大尺寸物块) ##########################
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
ee_link = franka.get_link('hand')

# 增大尺寸到 0.06，设定 Rough 表面和高摩擦力
def create_cube(pos, color):
    return scene.add_entity(
        gs.morphs.Box(
            size=(0.06, 0.06, 0.06), 
            pos=pos
        ),
        surface=gs.surfaces.Rough(color=color)
    )
# 定义一个更鲁棒的平滑移动函数
def move_to_smooth(goal_qpos, steps=100):
    if goal_qpos is None: return
    start_qpos = franka.get_dofs_position()
    # 简单的线性插值 (Joint Space Interpolation)
    for i in range(steps):
        t = (i + 1) / steps
        interp_q = start_qpos + (goal_qpos - start_qpos) * t
        franka.control_dofs_position(interp_q[:-2], motors_dof)
        # 维持夹爪握力
        franka.control_dofs_force(np.array([-15.0, -15.0]), fingers_dof)
        scene.step()

# 随机生成 4 个物块，两种颜色
cube_list = []
colors = [(1, 0, 0), (0, 0, 1)]  # 只有红(0)和蓝(1)
targets = [np.array([0.3, -0.4, 0.1]), np.array([0.3, 0.4, 0.1])] # 对应两个分拣位

print(">>> 正在生成 4 个随机物块...")

# 随机生成 4 个物块时，初始位置设为地下 (Z = -1.0)
for i in range(4):
    x = 0.45 + (i * 0.08)
    y = random.uniform(-0.2, 0.2)
    color_idx = i % 2
    color = colors[color_idx]
    
    # 存下原本想要的坐标，但 set_pos 先放地下
    cube_list.append({
        "entity": create_cube((x, y, -1.0), color), # 修改这里：暂时藏在地下
        "real_x": x, # 新增记录：原本的 X
        "real_y": y, # 新增记录：原本的 Y
        "color_type": color_idx,
        "target_pos": targets[color_idx],
        "size": 0.06 
    })

# 在 build() 之前定义堆叠管理器
# 键是目标的坐标（转为 tuple 方便寻址），值是当前已堆叠的数量
stack_counts = {tuple(target.tolist()): 0 for target in targets}

# 物块的基础厚度
CUBE_HEIGHT = 0.06

########################## build ##########################
scene.build()

# --- 新增步骤：机械臂先回到安全待命姿态 ---
print(">>> 机械臂正在前往待命姿态...")
# 这是一个经典的待命角度 (手肘弯曲，远离桌面中心)
standby_q = np.array([0, -0.78, 0, -2.35, 0, 1.57, 0.78, 0.04, 0.04])
franka.set_dofs_position(standby_q)

# 运行一段时间让机械臂完全静止
for _ in range(100):
    scene.step()

# --- 新增步骤：现在把物块“变”到桌面上 ---
print(">>> 机械臂已就位，物块投放中...")
for item in cube_list:
    # 将物块从地下移回桌面随机位置
    item["entity"].set_pos(np.array([item["real_x"], item["real_y"], 0.03]))

# 让方块自然沉降和稳定，此时机械臂已经是待命姿态，不会碰到它们
for _ in range(50):
    scene.step()

########################## 3. 增强版动作函数 ##########################
motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# 调回官方推荐的增益（手指 KP 降回 100，防止用力过猛抖动）
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))

def move_to(pos, num_waypoints=50):
    qpos = franka.inverse_kinematics(link=ee_link, pos=pos, quat=np.array([0, 1, 0, 0]))
    if qpos is not None:
        # 运动规划时，手指位置保持现状
        current_fingers = franka.get_dofs_position()[-2:]
        qpos[-2:] = current_fingers
        
        path = franka.plan_path(qpos_goal=qpos, num_waypoints=num_waypoints)
        if path is not None:
            for waypoint in path:
                franka.control_dofs_position(waypoint)
                scene.step()
    # 官方关键：动作完后给 100 步让 PD 控制器收敛
    for _ in range(100): scene.step()
    return qpos # 返回 qpos 供后续锁定手臂使用

########################## 4. 自动分拣主循环 ##########################
print(">>> 严格遵循官方逻辑的分拣程序启动...")



for i, item in enumerate(cube_list):
    cube = item["entity"]
    target = item["target_pos"]

    # 【强制重置】在处理每个物块前，必须强制切换回位置控制并张开手指
    # 这能解决你提到的“手指闭合去戳下一个物块”的问题
    temp_q = franka.get_dofs_position()
    temp_q[-2:] = 0.04
    franka.control_dofs_position(temp_q)
    for _ in range(50): scene.step()
    
    try:
        curr_pos = cube.get_pos().cpu().numpy()
        print(f"正在处理第 {i+1} 个物块...")
        
        # 1. 移动到上方并张开（确保下探前夹爪是开着的）
        print(">>> 步骤1：张开并移动到物体上方")
        move_to(curr_pos + [0, 0, 0.25])
        temp_q = franka.get_dofs_position()
        temp_q[-2:] = 0.04 # 显式张开
        franka.control_dofs_position(temp_q)
        for _ in range(100): scene.step() 
        
        # 2. 下探 (Reach)
        print(">>> 步骤2：下探到抓取高度")
        # 0.11 是偏置。如果你的物块 size 是 0.04，中心在 0.02，那么目标是 0.13
        # 如果物块 size 是 0.06，中心在 0.03，那么目标是 0.14
        grasp_height = 0.095 + (0.03 if item.get('size', 0.04) == 0.06 else 0.02)
        qpos_reach = franka.inverse_kinematics(link=ee_link, pos=curr_pos + [0, 0, grasp_height], quat=np.array([0, 1, 0, 0]))
        
        # 仅控制手臂下探，保持手指张开
        franka.control_dofs_position(qpos_reach[:-2], motors_dof)
        for _ in range(100): scene.step()

        # 3. 抓取 (Grasp) —— 【关键：你漏掉的步骤】
        print(">>> 步骤3：合拢夹爪并施加压力")
        # 锁定手臂当前位置，给手指施加持续向内的力
        franka.control_dofs_position(qpos_reach[:-2], motors_dof)
        franka.control_dofs_force(np.array([-10.0, -10.0]), fingers_dof) # 适当加大力度
        for _ in range(100): scene.step() 

        # 4. 提起 (Lift)
        print(">>> 步骤4：垂直提起")
        qpos_lift = franka.inverse_kinematics(link=ee_link, pos=curr_pos + [0, 0, 0.3], quat=np.array([0, 1, 0, 0]))
        if qpos_lift is not None:
            path = franka.plan_path(qpos_goal=qpos_lift, num_waypoints=100)
            for waypoint in path:
                # 每一帧都要：锁定手臂位置 + 维持手指握力
                franka.control_dofs_position(waypoint[:-2], motors_dof)
                franka.control_dofs_force(np.array([-10.0, -10.0]), fingers_dof)
                scene.step()
        # --- 步骤 5：转移到堆叠目标位 ---
        print(f">>> 步骤5：移动到分拣区域进行堆叠")
        
        # 获取当前目标点已经堆了几个
        current_stack_num = stack_counts[tuple(target.tolist())]
        
        # 计算本次放置的精确高度
        # 基础中心高度是 0.03，每多一个就加 0.06
        place_z = 0.03 + (current_stack_num * CUBE_HEIGHT)
        
        # 5.1 移动到堆叠点正上方 (hover_pos)
        hover_pos = np.array([target[0], target[1], place_z + 0.15])
        qpos_hover = franka.inverse_kinematics(link=ee_link, pos=hover_pos, quat=np.array([0, 1, 0, 0]))
        move_to_smooth(qpos_hover, steps=120)

        # 5.2 垂直缓慢下探到放置位 (place_pos)
        place_pos = np.array([target[0], target[1], place_z + 0.11 + 0.005])
        qpos_place = franka.inverse_kinematics(link=ee_link, pos=place_pos, quat=np.array([0, 1, 0, 0]))
        move_to_smooth(qpos_place, steps=60)

        # --- 步骤 6：精准释放 ---
        print(">>> 步骤6：触底释放")
        target_q = franka.get_dofs_position()
        target_q[-2:] = 0.04 # 张开
        franka.control_dofs_position(target_q)
        # 释放后多停留一会儿，让物块稳在堆叠塔上
        for _ in range(150): scene.step() 
        
        # 更新该位置的堆叠计数
        stack_counts[tuple(target.tolist())] += 1

        # 6.1 垂直抬起（防止张开的手指扫倒刚堆好的塔）
        lift_pos = place_pos + [0, 0, 0.1]
        qpos_post_lift = franka.inverse_kinematics(link=ee_link, pos=lift_pos, quat=np.array([0, 1, 0, 0]))
        move_to(lift_pos, num_waypoints=50)

        # 7. 彻底重置 (Reset)
        # 这步放在最后，为下一个物块做准备
        print(">>> 步骤7：完成分拣，重置状态")
        move_to(np.array([0.4, 0, 0.4]))

    except Exception as e:
        print(f"分拣出错: {e}")
