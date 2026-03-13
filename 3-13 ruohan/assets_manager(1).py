import genesis as gs
import numpy as np
import os
import glob

def create_shared_assets():
    """生成两种模式通用的 obj 模型"""
    if not os.path.exists("cube_final.obj"):
        obj_content = """
v -0.5 -0.5 0.5\nv 0.5 -0.5 0.5\nv 0.5 0.5 0.5\nv -0.5 0.5 0.5
v -0.5 -0.5 -0.5\nv 0.5 -0.5 -0.5\nv 0.5 0.5 -0.5\nv -0.5 0.5 -0.5
vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1
vn 0 0 1\nvn 0 0 -1\vn 0 1 0\vn 0 -1 0\vn 1 0 0\vn -1 0 0
f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\nf 8/1/2 7/2/2 6/3/2
f 8/1/2 6/3/2 5/4/2\nf 4/1/3 3/2/3 7/3/3\nf 4/1/3 7/3/3 8/4/3
f 5/1/4 1/2/4 4/3/4\nf 5/1/4 4/3/4 8/4/4\nf 5/1/5 6/2/5 2/3/5
f 5/1/5 2/3/5 1/4/5\nf 2/1/6 6/2/6 7/3/6\nf 2/1/6 7/3/6 3/4/6
"""
        with open("cube_final.obj", "w") as f: f.write(obj_content.strip())

def clear_temp_urdfs():
    """清理堆叠模式产生的临时文件"""
    files = glob.glob("box_L*.urdf")
    for f in files:
        try:
            os.remove(f)
            print(f"清理临时文件: {f}")
        except OSError:
            pass
    print("✨ 临时 URDF 清理完毕。")

def _spawn_scattered_boxes(scene, texture_list, count):
    """逻辑 A: 随机散落箱子 (原 assets_manager.py)"""
    if not os.path.exists("box_target.urdf"):
        urdf_content = """
<robot name="box">
  <link name="base">
    <inertial>
      <mass value="0.5000"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
    <visual><geometry><mesh filename="cube_final.obj" scale="0.12 0.12 0.12"/></geometry></visual>
    <collision><geometry><box size="0.12 0.12 0.12"/></geometry></collision>
  </link>
</robot>"""
        with open("box_target.urdf", "w") as f: f.write(urdf_content.strip())

    objs = []
    spawned_positions = [] 
    
    min_dist_to_base = 0.40   
    max_dist_to_base = 0.65   
    min_box_spacing = 0.15    
    max_attempts = 100        

    for i in range(count):
        found_valid_pos = False
        attempts = 0
        while not found_valid_pos and attempts < max_attempts:
            attempts += 1
            cand_x = np.random.uniform(0.40, 0.65)
            cand_y = np.random.uniform(-0.35, 0.35)
            dist_to_base = np.sqrt(cand_x**2 + cand_y**2)
            if not (min_dist_to_base <= dist_to_base <= max_dist_to_base):
                continue
            
            is_conflicting = False
            for (prev_x, prev_y) in spawned_positions:
                dist_to_prev = np.sqrt((cand_x - prev_x)**2 + (cand_y - prev_y)**2)
                if dist_to_prev < min_box_spacing:
                    is_conflicting = True
                    break
            if not is_conflicting:
                found_valid_pos = True
                spawned_positions.append((cand_x, cand_y))
        
        if not found_valid_pos: continue

        rand_mu = np.random.uniform(0.2, 0.9) 
        rand_scale = np.random.uniform(0.85, 1.0)
        chosen_tex = np.random.choice(texture_list)
        pos_z = 0.06 * rand_scale + 0.005
        
        obj = scene.add_entity(
            morph=gs.morphs.URDF(file='box_target.urdf', pos=(cand_x, cand_y, pos_z), scale=rand_scale),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(image_path=chosen_tex)
            )
        )
        
        obj.friction = rand_mu 
        objs.append(obj)
        print(f"📦 散落箱子 {i+1} 生成: ({cand_x:.2f}, {cand_y:.2f}) | 摩擦系数 μ={rand_mu:.2f}")
        
    return objs

def _spawn_stacked_boxes(scene, texture_list, count):
    """逻辑 B: 堆叠箱子 (原 assets_manager_1.py)"""
    center_x = np.random.uniform(0.48, 0.52)
    center_y = np.random.uniform(-0.05, 0.05)
    
    current_z_floor = 0.0 
    temp_obj_list = []
    boxes_per_layer = 2 

    for layer in range(count):
        layer_max_h = 0
        is_rotated = (layer % 2 == 1)
        
        for i in range(boxes_per_layer):
            sx = np.random.uniform(0.08, 0.11)
            sy = np.random.uniform(0.14, 0.17)
            sz = np.random.uniform(0.07, 0.10) 
            
            if is_rotated: sx, sy = sy, sx
            
            offset_val = 0.06 
            dx = (offset_val if i == 0 else -offset_val) if not is_rotated else 0
            dy = (offset_val if i == 0 else -offset_val) if is_rotated else 0
            
            gap = 0.003 
            pos_z = current_z_floor + (sz / 2.0) + gap
            
            urdf_idx = layer * boxes_per_layer + i
            urdf_filename = f"box_L{layer}_{i}.urdf"
            urdf_content = f"""
<robot name="box_{urdf_idx}">
  <link name="base">
    <inertial>
      <mass value="0.4"/> <inertia ixx="0.0005" ixy="0" ixz="0" iyy="0.0005" iyz="0" izz="0.0005"/>
    </inertial>
    <visual><geometry><mesh filename="cube_final.obj" scale="{sx} {sy} {sz}"/></geometry></visual>
    <collision><geometry><box size="{sx} {sy} {sz}"/></geometry></collision>
  </link>
</robot>"""
            with open(urdf_filename, "w") as f: f.write(urdf_content.strip())
            
            rand_mu = np.random.uniform(0.7, 0.9)
            chosen_tex = np.random.choice(texture_list) if texture_list else None
            
            obj = scene.add_entity(
                morph=gs.morphs.URDF(file=urdf_filename, pos=(center_x + dx, center_y + dy, pos_z), scale=1.0),
                surface=gs.surfaces.Default(
                    diffuse_texture=gs.textures.ImageTexture(image_path=chosen_tex) if chosen_tex else None
                )
            )
            
            obj.friction = rand_mu 
            temp_obj_list.append(obj)
            layer_max_h = max(layer_max_h, sz)

        current_z_floor += layer_max_h + gap
        print(f"📦 堆叠层 {layer+1} 已放置. 当前总高约为: {current_z_floor:.3f}m")

    return temp_obj_list[::-1]

def spawn_multiple_boxes(scene, texture_list, count=3, is_stacked=False):
    """
    统一入口函数。
    通过 is_stacked 参数控制生成逻辑。
    """
    create_shared_assets()
    
    if is_stacked:
        print("🔧 启用 [堆叠模式] 生成场景...")
        return _spawn_stacked_boxes(scene, texture_list, count)
    else:
        print("🔧 启用 [散落模式] 生成场景...")
        return _spawn_scattered_boxes(scene, texture_list, count)
