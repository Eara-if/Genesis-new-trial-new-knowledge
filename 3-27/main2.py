import numpy as np
import genesis as gs
from smart_gripper import SmartParallelGripper
from assets_manager import spawn_random_boxes
import cv2
import os

########################## 1. 初始化 ##########################
gs.init(backend=gs.gpu, precision='32')
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.005, substeps=10),
    show_viewer=True,  # 【修改】开启原生查看器
    viewer_options=gs.options.ViewerOptions(
        camera_fov=100.0,  # 设置广角视场角 (推荐 90.0 - 110.0，数值越大越广)
    ),
    renderer=gs.renderers.Rasterizer()
)

########################## 2. 实体加载 ##########################
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
ee_link = franka.get_link('hand')
motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

cube_list, texture_names = spawn_random_boxes(scene, count=4)
stack_heights = {tex: 0.03 for tex in texture_names}

# --- 新增：添加静态障碍物 ---
# 假设在抓取区(x≈0.45)和放置区(x=0.3)之间放置一堵小墙或一个柱子
# 1. 左侧障碍物（挡住去往 y=0.4 的路径）
obstacle_left = scene.add_entity(
    gs.morphs.Box(
        size=(0.3, 0.02, 0.3),   # 厚5cm, 宽20cm, 高15cm
        pos=(0.5, 0.3, 0.15)    # y=0.3 刚好挡在去往左侧分拣区的路上
    ),
    surface=gs.surfaces.Default(color=(0.8, 0.2, 0.2)) 
)
obstacle_right = scene.add_entity(
    gs.morphs.Box(
        size=(0.3, 0.02, 0.5), 
        pos=(0.2, 0.3, 0.25)   # y=0.3 刚好挡在去往右侧分拣区的路上
    ),
    surface=gs.surfaces.Default(color=(0.8, 0.2, 0.2))
)

# 2. 右侧障碍物（挡住去往 y=-0.4 的路径）
obstacle_right = scene.add_entity(
    gs.morphs.Box(
        size=(0.3, 0.02, 0.3), 
        pos=(0.5, -0.3, 0.15)   # y=-0.3 刚好挡在去往右侧分拣区的路上
    ),
    surface=gs.surfaces.Default(color=(0.8, 0.2, 0.2))
)

obstacle_right = scene.add_entity(
    gs.morphs.Box(
        size=(0.3, 0.02, 0.15), 
        pos=(0.2, -0.3, 0.75)   # y=-0.3 刚好挡在去往右侧分拣区的路上
    ),
    surface=gs.surfaces.Default(color=(0.8, 0.2, 0.2))
)


# Genesis 会自动将加入 scene 的 entity 纳入物理引擎和路径规划的碰撞检测中

########################## 3. 视角绑定逻辑 (修改部分) ##########################

def update_camera_display():
    """
    不再使用 OpenCV 窗口，直接更新原生 Viewer 的相机位置
    实现‘第一人称俯视’效果
    """
    if scene.viewer is not None:
        try:
            # 1. 获取 ee_link 的实时位姿
            pos = ee_link.get_pos().cpu().numpy()
            q = ee_link.get_quat().cpu().numpy()
            
            # 2. 计算局部轴的世界向量
            v_z = np.array([
                2*(q[1]*q[3] + q[0]*q[2]), 
                2*(q[2]*q[3] - q[0]*q[1]), 
                1 - 2*(q[1]**2 + q[2]**2)
            ])
            v_x = np.array([
                1 - 2*(q[2]**2 + q[3]**2), 
                2*(q[1]*q[2] + q[0]*q[3]), 
                2*(q[1]*q[3] - q[0]*q[2])
            ])

            # 3. 设置相机位置 (后退 0.15m，抬高 0.2m 形成俯视)
            cam_pos = pos - v_z * 0.1 + v_x * 0.2
            cam_lookat = pos + v_z * 0.2
            
            # 4. 直接同步给 Viewer
            scene.viewer.set_camera_pose(pos=cam_pos, lookat=cam_lookat)
            
        except Exception:
            pass

# 【修改】删除了原有的 hand_camera 传感器定义和 VideoWriter 初始化
# 因为我们现在直接观察 Viewer 窗口

########################## 5. 动作函数 ##########################

def move_to_hybrid(pos, num_waypoints=60, force=0.0, item=None):
    # 1. 传入当前姿态作为 IK 参考，防止角度跳变（抽风的核心诱因）
    current_qpos = franka.get_dofs_position()
    q_goal = franka.inverse_kinematics(
        link=ee_link, 
        pos=pos, 
        quat=np.array([0, 1, 0, 0]),
        # seed_qpos=current_qpos # 某些版本 Genesis 支持 seed 传入
    )
    
    if q_goal is not None:
        q_goal[-2:] = current_qpos[-2:] 
        
        # 2. 尝试规划路径
        path = franka.plan_path(qpos_goal=q_goal, num_waypoints=num_waypoints)
        
        if path is not None:
            # --- 抽风防护逻辑 ---
            # 检查起始点和路径第一帧的差距，如果单关节跳变 > 1.5弧度，说明 IK 解选错了
            if np.any(np.abs(path[0].cpu().numpy() - current_qpos.cpu().numpy()) > 1.5):
                print("⚠️ 检测到关节跳变，正在尝试平滑过渡...")
                joint_linear_step(q_goal, steps=100, force=force) # 降级为线性插值，更稳
                return

            for idx, waypoint in enumerate(path):
                franka.control_dofs_position(waypoint[:-2], motors_dof)
                franka.control_dofs_force(np.array([force, force]), fingers_dof)
                scene.step()
                update_camera_display()
                if item is not None and idx % 30 == 0:
                    ee_vel = ee_link.get_vel().cpu().numpy()[:3]
                    smart_gripper.print_status(idx, item['mass'], item['mu'], abs(force), abs(force)/1.5, ee_vel)
        else:
            print(f"⚠️ 路径规划被封死: {pos}")
    
    for _ in range(60):
        franka.control_dofs_force(np.array([force, force]), fingers_dof)
        scene.step()
        update_camera_display()


def joint_linear_step(goal_qpos, steps=80, force=0.0):
    start_qpos = franka.get_dofs_position().cpu().numpy()
    if hasattr(goal_qpos, 'cpu'):
        goal_qpos = goal_qpos.cpu().numpy()
    
    for i in range(steps):
        t = (i + 1) / steps
        interp_q = start_qpos + (goal_qpos - start_qpos) * t
        franka.control_dofs_position(interp_q[:-2], motors_dof)
        franka.control_dofs_force(np.array([force, force]), fingers_dof)
        scene.step()
        update_camera_display()
        
    for _ in range(50): 
        scene.step()
        update_camera_display()

########################## build ##########################
scene.build()

for _ in range(5):
    scene.step()

for item in cube_list:
    item["entity"].geoms[0].set_friction(item["mu"])

smart_gripper = SmartParallelGripper(franka, fingers_dof)
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))

jnt_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'finger_joint1', 'finger_joint2']
dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

standby_q = np.array([0, -0.78, 0, -2.35, 0, 1.57, 0.78, 0.045, 0.045])
franka.set_dofs_position(standby_q, dofs_idx)

for _ in range(100):
    franka.control_dofs_position(standby_q, dofs_idx)
    scene.step()
    update_camera_display()

for item in cube_list:
    # 初始高度设为高度的一半，确保刚好贴地
    init_z = item['dims'][2] / 2.0
    item["entity"].set_pos(np.array([item["real_x"], item["real_y"], init_z]))
for _ in range(50): 
    scene.step()
    update_camera_display()

########################## 6. 自动分拣主循环 ##########################
for i, item in enumerate(cube_list):
    cube = item["entity"]
    target = item["target_pos"]
    curr_pos = cube.get_pos().cpu().numpy()
    
    move_to_hybrid(curr_pos + [0, 0, 0.25], force=0.0)
    # 将原来的：
    # grasp_z = item['size'] / 2.0 + 0.11
    # 修改为（使用 dims 中的高度 side_h）：
    grasp_z = item['dims'][2] / 2.0 + 0.11
    q_down = franka.inverse_kinematics(link=ee_link, pos=curr_pos[:2].tolist() + [grasp_z], quat=np.array([0, 1, 0, 0]))
    
    if q_down is not None:
        open_width = 0.04
        current_q = franka.get_dofs_position()
        current_q[-2:] = open_width
        franka.control_dofs_position(current_q[:-2], motors_dof)
        franka.control_dofs_position(np.array([open_width, open_width]), fingers_dof)
        for _ in range(40): 
            scene.step()
            update_camera_display()

        q_down[-2:] = open_width 
        joint_linear_step(q_down, steps=60, force=0.0)

    required_f = smart_gripper.compute_required_force(item['mass'], item['mu'])
    current_applied_force = -max(required_f * 1.8, 15.0) 
    franka.control_dofs_force(np.array([current_applied_force, current_applied_force]), fingers_dof)
    for _ in range(120): 
        scene.step()
        update_camera_display()

    # ---------------- 原始代码 ----------------
    # move_to_hybrid(curr_pos + [0, 0, 0.35], force=current_applied_force, item=item)
    # move_to_hybrid(np.array([target[0], target[1], 0.35]), force=current_applied_force, item=item)
    
    # ---------------- 改造后的避障逻辑 ----------------
    
    # 4.1 垂直拔起：不要用避障规划，用线性插值直接拔高到安全高度
    # 既然障碍物高 0.3m，我们直接拔高到 0.45m，确保绝对安全
    safe_z_all = 0.48 
    q_lift = franka.inverse_kinematics(link=ee_link, pos=[curr_pos[0], curr_pos[1], safe_z_all], quat=np.array([0, 1, 0, 0]))
    if q_lift is not None:
        q_lift[-2:] = franka.get_dofs_position()[-2:]
        joint_linear_step(q_lift, steps=80, force=current_applied_force)

    # 4.2 跨越平移：此时已经在空中，调用 move_to_hybrid 让它自动绕过或飞过
    # 删掉手动设置的 via_point，因为它可能会把机械臂带向障碍物更近的地方
    # 直接让它去目标点上方
    print(">>> 正在跨越障碍物...")
    move_to_hybrid(np.array([target[0], target[1], safe_z_all]), force=current_applied_force, item=item)

    # 4.3 移动到目标点正上方
    move_to_hybrid(np.array([target[0], target[1], safe_z_all]), force=current_applied_force, item=item)

    
    place_z = stack_heights[item["texture"]] + (item["dims"][2] / 2.0)
    place_pos = np.array([target[0], target[1], place_z + 0.115 + 0.01])
    q_place = franka.inverse_kinematics(link=ee_link, pos=place_pos, quat=np.array([0, 1, 0, 0]))
    joint_linear_step(q_place, steps=100, force=current_applied_force)


    temp_q = franka.get_dofs_position()
    temp_q[-2:] = 0.04 
    franka.control_dofs_position(temp_q)
    for _ in range(150): 
        scene.step()
        update_camera_display()
    
    # 循环结束时的累加：
    stack_heights[item["texture"]] += item['dims'][2]
    # --- 安全回位逻辑 ---
    print(">>> 正在安全回位...")
    
    # 1. 原地垂直升空 (回到之前定义的绝对安全高度 0.48m)
    current_ee_pos = ee_link.get_pos().cpu().numpy()
    q_retract = franka.inverse_kinematics(
        link=ee_link, 
        pos=[current_ee_pos[0], current_ee_pos[1], safe_z_all], 
        quat=np.array([0, 1, 0, 0])
    )
    if q_retract is not None:
        joint_linear_step(q_retract, steps=60, force=0.0)

    # 2. 高空平移回到抓取区上方 (x=0.45, y=0, z=0.48)
    # 这样可以确保横向移动时，手爪在墙体正上方掠过
    move_to_hybrid(np.array([0.45, 0.0, safe_z_all]), force=0.0)

    # 3. 最后再执行关节归位，此时已经远离障碍物
    joint_linear_step(standby_q, steps=80, force=0.0)

# 【修改】删除了 video_writer.release()，现在直接关闭预览即可
print(">>> 演示完成。")