import os
import numpy as np
import genesis as gs

def create_textured_box_assets():
    # 生成带 UV 的模型文件
    if not os.path.exists("cube_final.obj"):
        obj_content = "v -0.5 -0.5 0.5\nv 0.5 -0.5 0.5\nv 0.5 0.5 0.5\nv -0.5 0.5 0.5\nv -0.5 -0.5 -0.5\nv 0.5 -0.5 -0.5\nv 0.5 0.5 -0.5\nv -0.5 0.5 -0.5\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\nvn 0 0 -1\nvn 0 1 0\nvn 0 -1 0\nvn 1 0 0\nvn -1 0 0\nf 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\nf 8/1/2 7/2/2 6/3/2\nf 8/1/2 6/3/2 5/4/2\nf 4/1/3 3/2/3 7/3/3\nf 4/1/3 7/3/3 8/4/4\nf 5/1/4 1/2/4 4/3/4\nf 5/1/4 4/3/4 8/4/4\nf 5/1/5 6/2/5 2/3/5\nf 5/1/5 2/3/5 1/4/5\nf 2/1/6 6/2/6 7/3/6\nf 2/1/6 7/3/6 3/4/6"
        with open("cube_final.obj", "w") as f: f.write(obj_content)

    if not os.path.exists("textured_box.urdf"):
        urdf_content = """<robot name="box"><link name="base"><inertial><mass value="0.2"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial><visual><geometry><mesh filename="cube_final.obj" scale="0.05 0.05 0.05"/></geometry></visual><collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision></link></robot>"""
        with open("textured_box.urdf", "w") as f: f.write(urdf_content)

def spawn_random_boxes(scene, count=4):
    create_textured_box_assets()
    # --- 新增：定义物理限制 ---
    MAX_GRIPPER_WIDTH = 0.08  # 假设夹爪最大开合 8cm
    SAFETY_MARGIN = 0.01      # 留出 1cm 余量
    UPPER_BOUND = MAX_GRIPPER_WIDTH - SAFETY_MARGIN # 最终上限 0.07
    LOWER_BOUND = 0.03        # 最终下限 0.03
    
    BASE_SIZE = 0.05
    # 计算对应的 scale 范围
    # min_scale = 0.03 / 0.05 = 0.6
    # max_scale = 0.07 / 0.05 = 1.4
    min_scale = LOWER_BOUND / BASE_SIZE
    max_scale = UPPER_BOUND / BASE_SIZE

    textures = ['box_surface.jpg', 'box_surface2.jpg'] 
    targets = {
        'box_surface.jpg': np.array([0.3, -0.4, 0.1]), 
        'box_surface2.jpg': np.array([0.3, 0.4, 0.1])
    }
    
    cube_list = []
    BASE_SIZE = 0.05
    
    for i in range(count):
        x = 0.45 + (i * 0.08)
        y = np.random.uniform(-0.2, 0.2)
        tex_path = textures[i % 2]
        rand_scale = np.random.uniform(min_scale, max_scale)
        actual_size = BASE_SIZE * rand_scale
        rand_mu = np.random.uniform(0.3, 0.9)
        # --- 修改：使用计算出的显著缩放范围 ---

        entity = scene.add_entity(
            morph=gs.morphs.URDF(file='textured_box.urdf', pos=(x, y, -1.0), scale=rand_scale),
            surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=tex_path))
        )
        
        cube_list.append({
            "entity": entity, "real_x": x, "real_y": y, "texture": tex_path,
            "target_pos": targets[tex_path], "size": actual_size, "mu": rand_mu, "mass": 0.2 * (rand_scale**3)
        })
    
    # 核心要求：按尺寸从大到小排序
    cube_list.sort(key=lambda item: item["size"], reverse=True)
    return cube_list, textures
