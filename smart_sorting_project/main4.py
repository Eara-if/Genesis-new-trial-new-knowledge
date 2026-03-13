import numpy as np
import genesis as gs
from smart_gripper import SmartParallelGripper

########################## 1. 初始化 ##########################
gs.init(backend=gs.gpu, precision='32')
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.2, 1.8, 1.5), 
        camera_lookat=(0.6, 0, 0.4)
    ),
    show_viewer=True
)

########################## 2. 实体加载 ##########################
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
ee_link = franka.get_link('hand')

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# --- 手动构建一张桌子 (修正版) ---
# --- 调整桌子和物块位置 ---
# 将桌子中心挪远一点，避免顶住机械臂底座

table_center = np.array([0.95, 0.0, 0.0]) # 再离远 10cm
t_w, t_d, t_h = 0.6, 0.8, 0.6            # 高度抬升至 0.6m

# 重新计算桌板和桌腿
table = scene.add_entity(
    gs.morphs.Box(
        size=(t_w, t_d, 0.02), 
        pos=(table_center[0], table_center[1], t_h),
        fixed=True
    ),
    surface=gs.surfaces.Default(color=(0.4, 0.3, 0.2))
)
leg_size = 0.04

# 2. 生成四条腿 (直接加到场景中，固定住)
# 腿的位置计算
offsets = [
    (-t_w/2 + leg_size/2, -t_d/2 + leg_size/2),
    ( t_w/2 - leg_size/2, -t_d/2 + leg_size/2),
    (-t_w/2 + leg_size/2,  t_d/2 - leg_size/2),
    ( t_w/2 - leg_size/2,  t_d/2 - leg_size/2)
]

legs = []
for i, (off_x, off_y) in enumerate(offsets):
    leg = scene.add_entity(
        gs.morphs.Box(
            size=(leg_size, leg_size, t_h),
            pos=(table_center[0] + off_x, table_center[1] + off_y, t_h / 2),
            fixed=True
        ),
        surface=gs.surfaces.Default(color=(0.3, 0.2, 0.1))
    )
    legs.append(leg)
# 将小物块放在桌子“靠近机械臂”的边缘
# 将物块放在更靠近机械臂的一侧，确保关节伸展自然
cube_pos = table_center + np.array([-0.22, 0.0, 0.03])
cube = scene.add_entity(
    gs.morphs.Box(size=(0.06, 0.06, 0.04), pos=cube_pos),
    surface=gs.surfaces.Default(color=(0.1, 0.8, 0.1))
)

target_drop_pos = np.array([0.4, 0.5, 0.05])

scene.build()

########################## 3. AABB 自动避障算法 ##########################

def move_with_aabb_avoidance(target_pos, finger_pos=0.045, force=0.0, speed=0.02):
    # 增加 link4，覆盖更多手臂面积
    arm_links = [franka.get_link(name) for name in ['link4', 'link5', 'link6', 'link7', 'hand']]

    for step in range(2000):
        scene.clear_debug_objects()
        curr_ee_pos = ee_link.get_pos().cpu().numpy()
        dist_to_goal = np.linalg.norm(target_pos - curr_ee_pos)
        if dist_to_goal < 0.01: break 

        repulsive_vec = np.zeros(3)
        
        # 1. 获取桌板 AABB
        plate_aabb = table.get_AABB()
        p_min, p_max = plate_aabb[0].cpu().numpy(), plate_aabb[1].cpu().numpy()
        
        # 2. 获取所有桌腿的 AABB (假设我们之前把 legs 存进了列表)
        all_obstacles = [(p_min, p_max, "plate")] # 格式: (min, max, type)
        for l_ent in legs:
            l_aabb = l_ent.get_AABB()
            all_obstacles.append((l_aabb[0].cpu().numpy(), l_aabb[1].cpu().numpy(), "leg"))

        buffer = 0.12

        for a_link in arm_links:
            a_pos = a_link.get_pos().cpu().numpy()
            
            for o_min, o_max, o_type in all_obstacles:
                closest_p = np.maximum(o_min, np.minimum(a_pos, o_max))
                dist = np.linalg.norm(a_pos - closest_p)
                
                if dist < buffer:
                    push_dir = (a_pos - closest_p) / (dist + 1e-6)
                    
                    if o_type == "plate":
                        # 如果在桌板下方，产生向下的压制力（防止撞板）或向上的排斥力
                        if a_pos[2] < o_min[2]: # 在桌底
                            # 增加一个向下的轻微推力，确保机械臂低头进入
                            if a_pos[2] > o_min[2] - 0.05:
                                repulsive_vec += np.array([0, 0, -0.15])
                        else: # 在桌上
                            repulsive_vec += np.array([0, 0, 0.3 * (buffer-dist)/buffer])
                    
                    else: # 撞到桌腿
                        push_dir[2] = 0 # 腿只产生水平排斥
                        repulsive_vec += push_dir * (0.4 * (buffer-dist)/buffer)
                        scene.draw_debug_line(start=a_pos, end=a_pos + push_dir * 0.1, color=(1, 0, 0))

        # 3. 动态合成运动
        diff = target_pos - curr_ee_pos
        attraction = (diff / (np.linalg.norm(diff) + 1e-6)) * speed
        
        # 混合：排斥力在靠近障碍时权重增加
        move_dir = attraction + repulsive_vec
        
        if np.linalg.norm(move_dir) > speed:
            move_dir = move_dir / np.linalg.norm(move_dir) * speed
        
        q_goal = franka.inverse_kinematics(link=ee_link, pos=curr_ee_pos + move_dir, quat=np.array([0, 1, 0, 0]))
        if q_goal is not None:
            franka.control_dofs_position(q_goal[:7], motors_dof)
            
            # --- 修复夹取状态判定 ---
            if force == 0:
                # 正常移动模式：保持手指张开
                franka.control_dofs_position(np.array([finger_pos, finger_pos]), fingers_dof)
            else:
                # 抓取/携带模式：施加恒定握力
                franka.control_dofs_force(np.array([force, force]), fingers_dof)
        
        scene.step()


########################## 4. 物理感知型执行流程 ##########################

# 1. 物理参数初始化 (学习自稳健代码：动态计算握力)
item_mass = 0.05 
item_mu = 0.5
smart_gripper = SmartParallelGripper(franka, fingers_dof)
required_f = smart_gripper.compute_required_force(item_mass, item_mu)
f_val = -max(required_f * 1.8, 15.0) 

# 设置控制增益 (Panda 机械臂的推荐稳健参数)
franka.set_dofs_kp(np.array([5000]*7 + [100, 100]))
franka.set_dofs_kv(np.array([500]*7 + [10, 10]))

print(f">>> 物理感知启动: 目标质量 {item_mass}kg, 自动计算握力 {abs(f_val)}N")


# 确保在 move 之前重新获取最新的物块 AABB 
cube_aabb = cube.get_AABB()
c_min, c_max = cube_aabb[0].cpu().numpy(), cube_aabb[1].cpu().numpy()
cube_height = c_max[2] - c_min[2]

# 获取实时中心位置
curr_cube_pos = cube.get_pos().cpu().numpy()

# 核心修正：抓取点应该直接对准物块中心
# 之前可能是因为 curr_pos 变量被错误引用导致位置偏差
grasp_pos = np.array([curr_cube_pos[0], curr_cube_pos[1], c_min[2] + cube_height/2])

# 3. 执行分段动作 (整合 AABB 自动避障)
print(">>> 避障探入...")
# 预到达点：在物块中心上方 10cm，保持手指张开 (force=0)
move_with_aabb_avoidance(grasp_pos + [0, 0, 0.1], speed=0.03) 

print(">>> 物理对准...")
# 精准下降到抓取中心高度
move_with_aabb_avoidance(grasp_pos, speed=0.01) 

print(">>> 智能锁紧...")
# 核心夹取动作：停止位移，切换到力控模式合拢手指
for _ in range(120):
    franka.control_dofs_force(np.array([f_val, f_val]), fingers_dof)
    scene.step()

print(">>> 带着物块避障退出...")
# 学习重点：在移动函数中传入 force=f_val，确保退出过程中物块不掉落
# 同时 AABB 算法会实时计算排斥力，绕开桌腿
move_with_aabb_avoidance(np.array([0.4, 0.0, 0.4]), force=f_val, speed=0.02)

print(">>> 自动分拣放置...")
move_with_aabb_avoidance(target_drop_pos + [0, 0, 0.1], force=f_val)

# 4. 释放并安全撤离
print(">>> 任务完成，松开物块...")
franka.control_dofs_position(np.array([0.045, 0.045]), fingers_dof)
for _ in range(100): scene.step()
