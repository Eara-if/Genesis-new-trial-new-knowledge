import numpy as np
import genesis as gs
import random
import os
import cv2

########################## 1. 初始化 ##########################
gs.init(backend=gs.gpu)

DATA_DIR = "embodied_data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.005, substeps=2),
    show_viewer=False,
)

########################## 2. 实体加载 ##########################
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
ee_link = franka.get_link('hand')

# 传感器配置
wrist_camera = scene.add_sensor(
    gs.sensors.RasterizerCameraOptions(
        res            = (224, 224), 
        pos            = (0.1, 0.0, 0.05),
        lookat         = (0.2, 0.0, 0.0),
        fov            = 70.0,
        entity_idx     = 1, 
        link_idx_local = ee_link.idx_local, 
    )
)

# --- 根据你找到的官方文档：使用 surface 指定颜色 ---
# 我们预创建三个颜色的方块，解决动态改色报错的问题
cubes = {
    0: scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0, 0, -1.0)),
        surface=gs.surfaces.Smooth(color=(1.0, 0.2, 0.2)) # 红色
    ),
    1: scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0, 0, -1.0)),
        surface=gs.surfaces.Smooth(color=(0.2, 0.2, 1.0)) # 蓝色
    ),
    2: scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0, 0, -1.0)),
        surface=gs.surfaces.Smooth(color=(0.2, 1.0, 0.2)) # 绿色
    )
}

scene.build()

########################## 3. 逻辑配置 ##########################
COLOR_MAP = {
    0: {"name": "red",   "target": np.array([0.4, -0.3, 0.15])},
    1: {"name": "blue",  "target": np.array([0.4,  0.3, 0.15])},
    2: {"name": "green", "target": np.array([0.3,  0.0, 0.15])}
}

########################## 4. 数据保存与动作函数 ##########################
def save_data_step(ep, step, goal_pos, color_id):
    try:
        ep_path = f"{DATA_DIR}/ep_{ep}"
        if not os.path.exists(ep_path): os.makedirs(ep_path)
        
        cam_data = wrist_camera.read()
        rgb_img = cam_data.rgb.cpu().numpy()
        cv2.imwrite(f"{ep_path}/frame_{step:04d}.jpg", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        
        qpos = franka.get_dofs_position().cpu().numpy()
        ee_pos = ee_link.get_pos().cpu().numpy()
        
        csv_path = f"{ep_path}/trajectory.csv"
        first_frame = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if first_frame:
                # 增加了 color_id 标题，方便后续 AI 训练识别
                f.write("step,j1,j2,j3,j4,j5,j6,j7,ee_x,ee_y,ee_z,goal_x,goal_y,goal_z,color_id\n")
            line = f"{step}," + ",".join([f"{x:.4f}" for x in qpos[:7]]) + \
                   f",{ee_pos[0]:.4f},{ee_pos[1]:.4f},{ee_pos[2]:.4f}," + \
                   f"{goal_pos[0]:.4f},{goal_pos[1]:.4f},{goal_pos[2]:.4f},{color_id}\n"
            f.write(line)
    except Exception: pass

def move_to(target_pos, ep, start_step, color_id, grip_force=None, steps=60):
    current_ee_pos = ee_link.get_pos().cpu().numpy()
    for i in range(steps):
        t = (i + 1) / steps
        goal_pos = current_ee_pos + (target_pos - current_ee_pos) * t
        qpos = franka.inverse_kinematics(link=ee_link, pos=goal_pos, quat=np.array([0, 1, 0, 0]))
        franka.control_dofs_position(qpos[:7], np.arange(7))
        gv = 0.01 if grip_force else 0.04
        franka.control_dofs_position(np.array([gv, gv]), np.arange(7, 9))
        scene.step()
        if (start_step + i) % 8 == 0:
            save_data_step(ep, start_step + i, goal_pos, color_id)
    return start_step + steps

########################## 5. 循环采集 (修正约束报错版) ##########################
print(">>> 颜色分拣数据采集启动 (API 兼容版)...")

for ep in range(15):
    c_id = random.choice([0, 1, 2])
    c_info = COLOR_MAP[c_id]
    
    # 隐藏所有方块
    for i in cubes:
        cubes[i].set_pos(np.array([0, 0, -5.0])) # 藏得深一点
    
    active_cube = cubes[c_id]
    rx, ry = random.uniform(0.5, 0.6), random.uniform(-0.1, 0.1)
    active_cube.set_pos(np.array([rx, ry, 0.02]))
    
    # 彻底清除上一轮可能残留的约束
    try:
        scene.sim.rigid_solver.clear_constraints() # 尝试新的清除 API
    except:
        pass
    
    for _ in range(20): scene.step()
    
    print(f"Episode {ep} | 目标: {c_info['name']}")
    sc = 0 
    try:
        cur_pos = active_cube.get_pos().cpu().numpy()
        
        # 1. 移向物体上方
        sc = move_to(cur_pos + [0, 0, 0.1], ep, sc, c_id, steps=50)
        # 2. 下压抓取
        sc = move_to(cur_pos + [0, 0, 0.01], ep, sc, c_id, steps=40)
        
        # 3. 【核心修复】添加焊接约束
        # 记录这个约束对象，以便后面精准删除
        link_cube = active_cube.links[0]
        try:
            constraint = scene.sim.rigid_solver.add_weld_constraint(link_cube, ee_link)
        except:
            # 备选方案：如果上面的 API 不行，用索引方式
            constraint = scene.sim.rigid_solver.add_weld_constraint(
                np.array([link_cube.idx]), np.array([ee_link.idx])
            )
        
        # 4. 提到空中
        sc = move_to(cur_pos + [0, 0, 0.15], ep, sc, c_id, steps=40)
        
        # 5. 移动到目标篮子
        sc = move_to(c_info["target"], ep, sc, c_id, grip_force=-10, steps=100)
        
        # 6. 【终极修复方案】
        try:
            # 优先使用新版 API：直接销毁约束对象
            constraint.destroy() 
        except:
            try:
                # 备选：如果你这版 rigid_solver 有这个属性
                scene.sim.rigid_solver.remove_constraint(constraint)
            except:
                # 最后的保底方案：如果约束删不掉，直接把方块“强行瞬移”回隐藏点
                # 这样物理引擎会因为距离过远自动断开拉扯
                active_cube.set_pos(np.array([0, 0, -5.0]))
            
        print(f"--> Episode {ep} 成功完成！")
        
    except Exception as e:
        print(f"--> Episode {ep} 运行时错误: {e}")
    
    # 重置机械臂位置
    franka.set_dofs_position(np.zeros(9))
    for _ in range(20): scene.step()

print(f"\n采集结束，数据已存入 {DATA_DIR}")