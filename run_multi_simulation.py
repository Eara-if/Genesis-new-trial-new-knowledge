import genesis as gs
import numpy as np
from waypoint_multi_task import WaypointMultiPickTask
#from assets_manager import spawn_multiple_boxes
from assets_manager_1 import spawn_multiple_boxes, clear_temp_urdfs

def main():
    gs.init(backend=gs.vulkan)
    
    scene = gs.Scene(
        show_viewer=True,
        sim_options=gs.options.SimOptions(dt=0.01, substeps=50)
    )
    scene.add_entity(gs.morphs.Plane())
    
    # 1. 加载机械臂
    robot = scene.add_entity(gs.morphs.MJCF(file='xml/universal_robots_ur5e/ur5e.xml'))
    
    # 2. 创建 4 个小吸盘
    suction_cups = []
    for _ in range(4):
        cup = scene.add_entity(
            gs.morphs.Cylinder(radius=0.015, height=0.03),
            surface=gs.surfaces.Plastic(color=(0.1, 0.1, 0.1))
        )
        suction_cups.append(cup)
    
    # 3. 生成目标池（多个盒子）
    my_textures = ['box_1.jpg', 'box_2.jpg'] # 确保你本地有这些图片，或者改为默认颜色
    obj_list = spawn_multiple_boxes(scene, my_textures, count=3)
    
    # 4. 初始化多目标任务
    task = WaypointMultiPickTask(scene, robot, suction_cups, obj_list)
    scene.build()
    task.start()

    # 仿真循环
    try:
        for step in range(20000): # 增加步数以完成多个任务
            task.step()
            scene.step()
    finally:
        # 无论程序是否报错，最后都清理文件
        clear_temp_urdfs()

if __name__ == "__main__":
    main()