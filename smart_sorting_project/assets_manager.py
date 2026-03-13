import os
import numpy as np
import genesis as gs

def create_real_box_assets(box_id, dims):
    """
    动态生成长方体 OBJ 和配套 URDF 
    dims: (length, width, height)
    """
    l, w, h = dims
    # 顶点坐标 
    v = [
        [-l/2, -w/2, h/2], [l/2, -w/2, h/2], [l/2, w/2, h/2], [-l/2, w/2, h/2],
        [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2], [l/2, w/2, -h/2], [-l/2, w/2, -h/2]
    ]
    
    obj_content = ""
    for vert in v:
        obj_content += f"v {vert[0]} {vert[1]} {vert[2]}\n"
    
    # 基础 UV 映射 
    obj_content += "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
    obj_content += "vn 0 0 1\nvn 0 0 -1\nvn 0 1 0\nvn 0 -1 0\nvn 1 0 0\nvn -1 0 0\n"
    
    # 面索引 
    faces = [
        "f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1",
        "f 8/1/2 7/2/2 6/3/2\nf 8/1/2 6/3/2 5/4/2",
        "f 4/1/3 3/2/3 7/3/3\nf 4/1/3 7/3/3 8/4/3",
        "f 5/1/4 1/2/4 4/3/4\nf 5/1/4 4/3/4 8/4/4",
        "f 5/1/5 6/2/5 2/3/5\nf 5/1/5 2/3/5 1/4/5",
        "f 2/1/6 6/2/6 7/3/6\nf 2/1/6 7/3/6 3/4/6"
    ]
    obj_content += "\n".join(faces)
    
    obj_name = f"box_{box_id}.obj"
    with open(obj_name, "w") as f: f.write(obj_content)
    
    # 动态生成 URDF，碰撞体(collision)尺寸与视觉(visual)一致 
    urdf_name = f"box_{box_id}.urdf"
    urdf_content = f"""<robot name="box_{box_id}">
        <link name="base">
            <inertial>
                <mass value="0.2"/>
                <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
            </inertial>
            <visual>
                <geometry><mesh filename="{obj_name}" scale="1 1 1"/></geometry>
            </visual>
            <collision>
                <geometry><box size="{l} {w} {h}"/></geometry>
            </collision>
        </link></robot>"""
    with open(urdf_name, "w") as f: f.write(urdf_content)
    return urdf_name

def spawn_random_boxes(scene, count=4):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    textures_names = ['box_surface.jpg', 'box_surface2.jpg']
    textures = [os.path.join(current_dir, name) for name in textures_names]
    
    targets = {
        textures[0]: np.array([0.3, -0.55, 0.1]), 
        textures[1]: np.array([0.3, 0.55, 0.1])
    }
    
    MIN_GRIP, MAX_GRIP = 0.035, 0.050
    cube_list = []

    # --- 核心修改：生成不重叠的随机坐标 ---
    # 将 0.5 到 0.75 分成 count 份
    x_bins = np.linspace(0.5, 0.75, count + 1) 
    
    # 随机化顺序，防止大尺寸物体总是出现在同一侧
    indices = list(range(count))
    np.random.shuffle(indices)

    for i in range(count):
        # 1. 尺寸随机
        side_a = np.random.uniform(MIN_GRIP, MAX_GRIP)
        side_b = np.random.uniform(MIN_GRIP, 0.08)
        side_h = np.random.uniform(0.02, 0.08)
        dims = (side_a, side_b, side_h)

        # 2. 坐标随机（在所属的 bin 范围内随机，并留出安全间距）
        bin_idx = indices[i]
        x_low = x_bins[bin_idx] + 0.02  # 留出 2cm 边界防止跨界重叠
        x_high = x_bins[bin_idx+1] - 0.02
        x = np.random.uniform(x_low, x_high)
        
        y = np.random.uniform(-0.18, 0.18) # 稍微收缩 Y 范围防止掉出桌子
        
        # 3. 创建与添加
        urdf_file = create_real_box_assets(i, dims)
        tex_path = textures[i % len(textures)]
        
        # 抬高 z 轴起始点，防止一半埋在地下 (side_h/2 是中心点高度)
        z_pos = side_h / 2.0 

        entity = scene.add_entity(
            morph=gs.morphs.URDF(file=urdf_file, pos=(x, y, z_pos)),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(image_path=tex_path),
                roughness=0.8
            )
        )
        
        cube_list.append({
            "entity": entity, "real_x": x, "real_y": y, 
            "texture": tex_path, "target_pos": targets[tex_path], 
            "size": side_a, "mu": np.random.uniform(0.3, 0.9), 
            "mass": 0.2, "dims": dims
        })
    
    cube_list.sort(key=lambda item: item["size"], reverse=True)
    return cube_list, textures
