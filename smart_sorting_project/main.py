import numpy as np
import genesis as gs
from smart_gripper import SmartParallelGripper
from assets_manager import spawn_random_boxes

########################## 1. 初始化 ##########################
gs.init(backend=gs.gpu)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.005, substeps=10),
    viewer_options=gs.options.ViewerOptions(camera_pos=(2.0, 1.5, 1.5), camera_lookat=(0.5, 0.0, 0.5), camera_fov=40),
    show_viewer=True,
)

########################## 2. 实体加载 ##########################
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
ee_link = franka.get_link('hand')
motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

cube_list, texture_names = spawn_random_boxes(scene, count=4)
stack_heights = {tex: 0.03 for tex in texture_names}

########################## 3. 动作函数 (完全保留原逻辑) ##########################
def move_to_hybrid(pos, num_waypoints=60, force=0.0, item=None):
    # 增加了一个关键参数：使用当前姿态作为初值，防止 IK 跳变到“立正”位
    q_goal = franka.inverse_kinematics(link=ee_link, pos=pos, quat=np.array([0, 1, 0, 0]))
    if q_goal is not None:
        current_q = franka.get_dofs_position()
        q_goal[-2:] = current_q[-2:]
        path = franka.plan_path(qpos_goal=q_goal, num_waypoints=num_waypoints)
        if path is not None:
            for idx, waypoint in enumerate(path):
                franka.control_dofs_position(waypoint[:-2], motors_dof)
                franka.control_dofs_force(np.array([force, force]), fingers_dof)
                scene.step()
                if item is not None and idx % 20 == 0:
                    ee_vel = ee_link.get_vel().cpu().numpy()[:3]
                    smart_gripper.print_status(idx, item['mass'], item['mu'], abs(force), abs(force)/1.5, ee_vel)
    for _ in range(80):
        franka.control_dofs_force(np.array([force, force]), fingers_dof)
        scene.step()

def joint_linear_step(goal_qpos, steps=80, force=0.0):
    """用于最后几厘米的垂直下探或放置，极其平稳"""
    # 核心修正：先 .cpu() 搬回内存，再 .numpy() 转换，最后确保 goal_qpos 也是 numpy 格式
    start_qpos = franka.get_dofs_position().cpu().numpy()
    
    # 如果 goal_qpos 是 Tensor 类型（比如 IK 返回的结果），也需要转换
    if hasattr(goal_qpos, 'cpu'):
        goal_qpos = goal_qpos.cpu().numpy()
    
    for i in range(steps):
        t = (i + 1) / steps
        interp_q = start_qpos + (goal_qpos - start_qpos) * t
        
        # 控制时 Genesis 会自动处理 numpy -> tensor 的转换
        franka.control_dofs_position(interp_q[:-2], motors_dof)
        franka.control_dofs_force(np.array([force, force]), fingers_dof)
        scene.step()
    for _ in range(50): scene.step()

########################## build ##########################
scene.build()

for item in cube_list:
    item["entity"].geoms[0].set_friction(item["mu"])

smart_gripper = SmartParallelGripper(franka, fingers_dof)
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))

# 获取官方风格的 DOF 索引映射
jnt_names = [
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7',
    'finger_joint1', 'finger_joint2',
]
dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]
motors_idx = dofs_idx[:7]  # 机械臂 7 轴
fingers_idx = dofs_idx[7:] # 夹爪 2 轴

# --- 官方风格的美化初始化 ---
print(">>> 机械臂正在平滑前往待命姿态...")
# 预设一个优雅的待命姿态
standby_q = np.array([0, -0.78, 0, -2.35, 0, 1.57, 0.78, 0.045, 0.045])

# 先进行一次硬重置（仅在仿真开始时）
franka.set_dofs_position(standby_q, dofs_idx)

# 关键：使用 PD 控制维持这个姿态，让物理系统稳定
for _ in range(100):
    franka.control_dofs_position(standby_q, dofs_idx)
    scene.step()

# --- 核心修正：先待命，后投放，防止扫到物块 ---
print(">>> 机械臂已就位，物块投放中...")
for item in cube_list:
    item["entity"].set_pos(np.array([item["real_x"], item["real_y"], 0.03]))
for _ in range(50): scene.step()

########################## 4. 自动分拣主循环 ##########################
for i, item in enumerate(cube_list):
    cube = item["entity"]
    target = item["target_pos"]
    
    print(f"\n>>> 正在处理: 尺寸 {item['size']*100:.1f}cm 的箱子")
    
    curr_pos = cube.get_pos().cpu().numpy()
    
    # 1. 移动到物块正上方（保持足够高的安全距离）
    move_to_hybrid(curr_pos + [0, 0, 0.25], force=0.0)

    # 2. 精准下探准备
    grasp_z = item['size'] / 2.0 + 0.11
    q_down = franka.inverse_kinematics(link=ee_link, pos=curr_pos[:2].tolist() + [grasp_z], quat=np.array([0, 1, 0, 0]))
    
    if q_down is not None:
        # --- 关键修正：先在上方张开手指 ---
        open_width = (item['size'] / 2.0) + 0.015 # 稍微多给点余量
        
        # 获取当前姿态并只修改手指部分
        current_q = franka.get_dofs_position()
        current_q[-2:] = open_width
        
        # 先执行张开动作（在当前位置）
        franka.control_dofs_position(current_q[:-2], motors_dof)
        franka.control_dofs_position(np.array([open_width, open_width]), fingers_dof)
        for _ in range(40): scene.step() # 等待手指张开

        # --- 然后再执行下探 ---
        q_down[-2:] = open_width # 确保下探过程中手指保持张开
        joint_linear_step(q_down, steps=60, force=0.0)

    # 3. 智能抓取
    required_f = smart_gripper.compute_required_force(item['mass'], item['mu'])
    # 增大基础握力，确保能抓起由于尺寸变大而变重的物体
    current_applied_force = -max(required_f * 1.8, 15.0) 
    franka.control_dofs_force(np.array([current_applied_force, current_applied_force]), fingers_dof)
    for _ in range(120): scene.step()

    # 4. 提升与平移
    move_to_hybrid(curr_pos + [0, 0, 0.35], force=current_applied_force, item=item)
    move_to_hybrid(np.array([target[0], target[1], 0.35]), force=current_applied_force, item=item)
    
    # 5. 放置
    place_z = stack_heights[item["texture"]] + (item["size"] / 2.0)
    place_pos = np.array([target[0], target[1], place_z + 0.115 + 0.01])
    q_place = franka.inverse_kinematics(link=ee_link, pos=place_pos, quat=np.array([0, 1, 0, 0]))
    joint_linear_step(q_place, steps=100, force=current_applied_force)

    # 6. 释放
    temp_q = franka.get_dofs_position()
    temp_q[-2:] = 0.04 
    franka.control_dofs_position(temp_q)
    for _ in range(150): scene.step()
    
    stack_heights[item["texture"]] += item["size"]

    # --- 核心修正：解决“立正”和“抽风”的关键 ---
    # 在抓取下一个之前，强制先执行一个关节空间的线性插值，回到你最帅的待命姿态
    print(">>> 正在归位，准备下一个目标...")
    joint_linear_step(standby_q, steps=100, force=0.0) 

print(">>> 所有物块分拣完毕！")