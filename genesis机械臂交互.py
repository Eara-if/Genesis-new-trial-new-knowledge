import numpy as np
import genesis as gs
import random
import pickle
import os
from datetime import datetime

########################## 1. 初始化 ##########################
gs.init(backend=gs.gpu)
trajectory_data = []

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.005, substeps=10),
    show_viewer=False, # 收集数据建议关闭，若需调试可改为 True
)

########################## 2. 实体与相机加载 ##########################
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
ee_link = franka.get_link('hand')

# 添加相机
cam = scene.add_camera(
    res    = (320, 240),
    pos    = (1.2, 0.0, 0.8),
    lookat = (0.5, 0.0, 0.2),
    fov    = 60,
)

def create_cube(pos, color):
    return scene.add_entity(
        gs.morphs.Box(size=(0.06, 0.06, 0.06), pos=pos),
        surface=gs.surfaces.Rough(color=color)
    )

def record_step():
    """镜像录制函数"""
    renders = cam.render()
    rgb = renders[0]
    qpos = franka.get_dofs_position().cpu().numpy()
    
    frame = {
        "observation": {
            "rgb": rgb,
            "qpos": qpos,
        },
        "action": qpos, 
    }
    trajectory_data.append(frame)

########################## 3. 动作函数 (完全镜像可视化逻辑) ##########################
motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

def move_to_smooth_collect(goal_qpos, steps=100):
    """带录制的 move_to_smooth"""
    if goal_qpos is None: return
    start_qpos = franka.get_dofs_position()
    for i in range(steps):
        t = (i + 1) / steps
        interp_q = start_qpos + (goal_qpos - start_qpos) * t
        franka.control_dofs_position(interp_q[:-2], motors_dof)
        # 严格遵循你的逻辑：维持握力
        franka.control_dofs_force(np.array([-15.0, -15.0]), fingers_dof)
        scene.step()
        record_step()

def move_to_collect(pos, num_waypoints=50):
    """带录制的 move_to (使用 plan_path)"""
    qpos = franka.inverse_kinematics(link=ee_link, pos=pos, quat=np.array([0, 1, 0, 0]))
    if qpos is not None:
        current_fingers = franka.get_dofs_position()[-2:]
        qpos[-2:] = current_fingers
        path = franka.plan_path(qpos_goal=qpos, num_waypoints=num_waypoints)
        if path is not None:
            for waypoint in path:
                franka.control_dofs_position(waypoint)
                scene.step()
                record_step()
    # 收敛步
    for _ in range(100): 
        scene.step()
        record_step()
    return qpos

########################## 4. 分拣主流程 ##########################

# 随机生成 4 个物块
cube_list = []
colors = [(1, 0, 0), (0, 0, 1)] 
targets = [np.array([0.3, -0.4, 0.1]), np.array([0.3, 0.4, 0.1])]
stack_counts = {tuple(target.tolist()): 0 for target in targets}
CUBE_HEIGHT = 0.06

for i in range(4):
    x, y = 0.45 + (i * 0.08), random.uniform(-0.2, 0.2)
    color_idx = i % 2
    cube_list.append({
        "entity": create_cube((x, y, -1.0), colors[color_idx]),
        "real_pos": np.array([x, y, 0.03]),
        "target_pos": targets[color_idx]
    })

scene.build()

# 设置初始姿态
standby_q = np.array([0, -0.78, 0, -2.35, 0, 1.57, 0.78, 0.04, 0.04])
franka.set_dofs_position(standby_q)
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))

for _ in range(100): scene.step()

# 放置物体并稳定
for item in cube_list:
    item["entity"].set_pos(item["real_pos"])
for _ in range(50): scene.step()

print(">>> 开始严格按照可视化逻辑录制...")

try:
    for i, item in enumerate(cube_list):
        cube = item["entity"]
        target = item["target_pos"]

        # 重置夹爪
        temp_q = franka.get_dofs_position()
        temp_q[-2:] = 0.04
        franka.control_dofs_position(temp_q)
        for _ in range(50): 
            scene.step()
            record_step()
        
        curr_pos = cube.get_pos().cpu().numpy()
        print(f"进程: [{i+1}/4] 处理中...")

        # 1. 移动到上方
        move_to_collect(curr_pos + [0, 0, 0.25])
        
        # 2. 显式张开手指
        temp_q = franka.get_dofs_position()
        temp_q[-2:] = 0.04
        franka.control_dofs_position(temp_q)
        for _ in range(100): 
            scene.step()
            record_step()

        # 3. 下探
        grasp_height = 0.095 + 0.03 # size 0.06 的中心偏置
        qpos_reach = franka.inverse_kinematics(link=ee_link, pos=curr_pos + [0, 0, grasp_height], quat=[0,1,0,0])
        franka.control_dofs_position(qpos_reach[:-2], motors_dof)
        for _ in range(100): 
            scene.step()
            record_step()

        # 4. 抓取 (合拢并施压)
        franka.control_dofs_position(qpos_reach[:-2], motors_dof)
        franka.control_dofs_force(np.array([-20.0, -20.0]), fingers_dof) # 稍微加大到20保证稳固
        for _ in range(100): 
            scene.step()
            record_step()

        # 5. 提起
        q_lift_pos = curr_pos + [0, 0, 0.3]
        qpos_lift = franka.inverse_kinematics(link=ee_link, pos=q_lift_pos, quat=[0,1,0,0])
        path = franka.plan_path(qpos_goal=qpos_lift, num_waypoints=100)
        for waypoint in path:
            franka.control_dofs_position(waypoint[:-2], motors_dof)
            franka.control_dofs_force(np.array([-20.0, -20.0]), fingers_dof)
            scene.step()
            record_step()

        # 6. 移动到堆叠位
        current_stack_num = stack_counts[tuple(target.tolist())]
        place_z = 0.03 + (current_stack_num * CUBE_HEIGHT)
        
        q_hover_stack = franka.inverse_kinematics(link=ee_link, pos=target + [0, 0, place_z + 0.15], quat=[0,1,0,0])
        move_to_smooth_collect(q_hover_stack, steps=120)

        q_place = franka.inverse_kinematics(link=ee_link, pos=target + [0, 0, place_z + 0.11 + 0.005], quat=[0,1,0,0])
        move_to_smooth_collect(q_place, steps=60)

        # 7. 释放
        target_q = franka.get_dofs_position()
        target_q[-2:] = 0.04
        franka.control_dofs_position(target_q)
        for _ in range(150): 
            scene.step()
            record_step()
        
        stack_counts[tuple(target.tolist())] += 1
        
        # 8. 抬起并重置
        move_to_collect(target + [0, 0, place_z + 0.25])
        move_to_collect(np.array([0.4, 0, 0.4]))

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"sorting_data_{timestamp}.pkl", "wb") as f:
        pickle.dump(trajectory_data, f)
    print(f"数据保存成功，总帧数: {len(trajectory_data)}")

except Exception as e:
    print(f"录制中断: {e}")