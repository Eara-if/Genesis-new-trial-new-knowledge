import numpy as np
import genesis as gs

class LocalPlanner:
    def __init__(self, scene, robot, obstacles):
        self.scene = scene
        self.robot = robot
        self.ee_link = robot.get_link('hand')
        self.arm_links = [robot.get_link(name) for name in ['link5', 'link6', 'link7', 'hand']]
        self.obstacles = obstacles
        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        
        # 定义力的显示比例（为了让你看清，把向量长度放大了）
        self.viz_scale = 1.5 

    def move_to(self, target_pos, finger_pos=0.045, force=0.0, speed=0.03, avoid_obstacles=True, debug_visual=True):
        downward_quat = np.array([0, 1, 0, 0])

        for step in range(1200):
            curr_ee_pos = self.ee_link.get_pos().cpu().numpy()
            dist_to_goal = np.linalg.norm(target_pos - curr_ee_pos)
            if dist_to_goal < 0.005: break 

            # --- 1. 计算引力 (Attraction) ---
            # 方向：当前点指向目标点
            diff = target_pos - curr_ee_pos
            attraction_dir = diff / (np.linalg.norm(diff) + 1e-6)
            # 引力大小受 speed 控制
            attraction_vec = attraction_dir * speed

            # --- 2. 计算斥力 (Repulsion) ---
            repulsive_vec = np.zeros(3)
            if avoid_obstacles:
                buffer = 0.12 # 斥力范围：12cm
                for link in self.arm_links:
                    l_pos = link.get_pos().cpu().numpy()
                    for obs in self.obstacles:
                        o_min, o_max = obs.get_AABB()[0].cpu().numpy(), obs.get_AABB()[1].cpu().numpy()
                        closest_p = np.maximum(o_min, np.minimum(l_pos, o_max))
                        dist = np.linalg.norm(l_pos - closest_p)
                        
                        if dist < buffer:
                            w = (buffer - dist) / buffer # 距离越近权重越大 (0~1)
                            push_dir = (l_pos - closest_p) / (dist + 1e-6)
                            
                            # 屏蔽向下斥力（防止桌板把手臂压向地面）
                            if push_dir[2] < 0: push_dir[2] = 0
                            
                            # 单个障碍物的斥力强度设为 0.25 * 权重
                            repulsive_vec += push_dir * 0.25 * w

            # --- 3. 计算合力 (Total Force/Velocity) ---
            move_dir = attraction_vec + repulsive_vec
            
            # 限制合力的最大步幅
            if np.linalg.norm(move_dir) > speed:
                move_dir = move_dir / np.linalg.norm(move_dir) * speed

            # --- 4. 具象化表现：在仿真界面画线 ---
            if debug_visual and step % 2 == 0:
                # 绿色：引力方向 (Attraction)
                self.scene.draw_debug_line(
                    start=curr_ee_pos, 
                    end=curr_ee_pos + attraction_vec * self.viz_scale, 
                    color=(0, 1, 0)
                )
                
                # 红色：总斥力方向 (Repulsion)
                self.scene.draw_debug_line(
                    start=curr_ee_pos, 
                    end=curr_ee_pos + repulsive_vec * self.viz_scale, 
                    color=(1, 0, 0)
                )
                
                # 蓝色：最终合力方向 (Total Force)
                self.scene.draw_debug_line(
                    start=curr_ee_pos, 
                    end=curr_ee_pos + move_dir * self.viz_scale, 
                    color=(0, 0, 1)
                )

            # --- 5. 执行控制 ---
            q_goal = self.robot.inverse_kinematics(link=self.ee_link, pos=curr_ee_pos + move_dir, quat=downward_quat)
            if q_goal is not None:
                self.robot.control_dofs_position(q_goal[:7], self.motors_dof)
                if force == 0:
                    self.robot.control_dofs_position(np.array([finger_pos, finger_pos]), self.fingers_dof)
                else:
                    self.robot.control_dofs_force(np.array([force, force]), self.fingers_dof)
            
            self.scene.step()