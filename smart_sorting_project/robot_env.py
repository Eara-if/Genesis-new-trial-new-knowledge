import numpy as np
import genesis as gs
from assets_manager import spawn_random_boxes

def setup_environment():
    gs.init(backend=gs.gpu, precision='32')
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.2, 1.8, 1.5), 
            camera_lookat=(0.6, 0, 0.4)
        ),
        show_viewer=True
    )

    plane = scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
    
    # --- 1. 构建桌子 ---
    table_center = np.array([1.0, 0.0, 0.0]) # 离基座稍远，给机械臂留出探入空间
    t_w, t_d, t_h = 0.6, 0.8, 0.55             # 调整桌高为 0.55m
    
    table_plate = scene.add_entity(
        gs.morphs.Box(size=(t_w, t_d, 0.02), pos=(table_center[0], table_center[1], t_h), fixed=True),
        surface=gs.surfaces.Default(color=(0.4, 0.3, 0.2))
    )
    
    # 生成桌腿
    leg_size = 0.04
    offsets = [
        (-t_w/2 + leg_size/2, -t_d/2 + leg_size/2),
        ( t_w/2 - leg_size/2, -t_d/2 + leg_size/2),
        (-t_w/2 + leg_size/2,  t_d/2 - leg_size/2),
        ( t_w/2 - leg_size/2,  t_d/2 - leg_size/2)
    ]
    legs = []
    for off_x, off_y in offsets:
        leg = scene.add_entity(
            gs.morphs.Box(size=(leg_size, leg_size, t_h), 
                         pos=(table_center[0] + off_x, table_center[1] + off_y, t_h / 2), 
                         fixed=True),
            surface=gs.surfaces.Default(color=(0.3, 0.2, 0.1))
        )
        legs.append(leg)
    
    # --- 2. 生成带纹理的随机物块 (位于桌子下方区域) ---
    # spawn_random_boxes 会返回实体列表及相关物理参数
    cube_list, _ = spawn_random_boxes(scene, count=4) 

    scene.build()
    
    # 设置机械臂控制增益
    franka.set_dofs_kp(np.array([5000]*7 + [100, 100]))
    franka.set_dofs_kv(np.array([500]*7 + [10, 10]))
    
    return scene, franka, table_plate, legs, cube_list