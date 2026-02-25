import genesis as gs
import numpy as np
import os
import glob

def create_assets():
    """生成标准立方体 OBJ 资源"""
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
    """扫描并删除所有生成的临时 URDF 文件"""
    files = glob.glob("box_L*.urdf")
    for f in files:
        try:
            os.remove(f)
            print(f"清理临时文件: {f}")
        except OSError:
            pass
    print("✨ 临时 URDF 清理完毕。")

def spawn_multiple_boxes(scene, texture_list, count=3):
    """
    生成三层紧凑且稳定的快递堆。
    缩减了尺寸以适配 SAFE_Z = 0.45m 的工作空间。
    """
    create_assets()
    
    # 堆栈中心位置
    center_x = np.random.uniform(0.48, 0.52)
    center_y = np.random.uniform(-0.05, 0.05)
    
    current_z_floor = 0.0 
    temp_obj_list = []
    boxes_per_layer = 2 

    for layer in range(count):
        layer_max_h = 0
        is_rotated = (layer % 2 == 1) # 奇偶层方向交错以增加稳定性
        
        for i in range(boxes_per_layer):
            # --- 尺寸调整：整体缩小以适配空间 ---
            # 缩减后的长宽更紧凑，高度 sz 控制在 0.07-0.10
            sx = np.random.uniform(0.08, 0.11)
            sy = np.random.uniform(0.14, 0.17)
            sz = np.random.uniform(0.07, 0.10) 
            
            if is_rotated: sx, sy = sy, sx
            
            # 排列偏置：减小间距使堆叠更紧密
            offset_val = 0.06 
            dx = (offset_val if i == 0 else -offset_val) if not is_rotated else 0
            dy = (offset_val if i == 0 else -offset_val) if is_rotated else 0
            
            gap = 0.003 # 减小物理间隙
            pos_z = current_z_floor + (sz / 2.0) + gap
            
            # 动态 URDF 生成
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
            
            rand_mu = np.random.uniform(0.7, 0.9) # 保持高摩擦
            chosen_tex = np.random.choice(texture_list) if texture_list else None
            
            obj = scene.add_entity(
                morph=gs.morphs.URDF(file=urdf_filename, pos=(center_x + dx, center_y + dy, pos_z), scale=1.0),
                surface=gs.surfaces.Default(
                    diffuse_texture=gs.textures.ImageTexture(image_path=chosen_tex) if chosen_tex else None
                )
            )
            
            obj.friction = rand_mu # 注入摩擦力
            temp_obj_list.append(obj)
            layer_max_h = max(layer_max_h, sz)

        current_z_floor += layer_max_h + gap
        print(f"📦 层 {layer+1} 已放置. 当前总高约为: {current_z_floor:.3f}m")

    # 逆序返回，确保机械臂从最顶层开始抓取
    return temp_obj_list[::-1]