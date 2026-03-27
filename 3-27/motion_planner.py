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

    def move_to(self, target_pos, finger_pos=0.045, force=0.0, speed=0.03, avoid_obstacles=True, callback=None):
        downward_quat = np.array([0, 1, 0, 0])

        for step in range(1200):
            curr_ee_pos = self.ee_link.get_pos().cpu().numpy()
            dist_to_goal = np.linalg.norm(target_pos - curr_ee_pos)
            if dist_to_goal < 0.005: 
                break 

            # --- 1. 计算引力 (Attraction) ---
            diff = target_pos - curr_ee_pos
            attraction_dir = diff / (np.linalg.norm(diff) + 1e-6)
            attraction_vec = attraction_dir * speed

            # --- 2. 计算斥力 (Repulsion) ---
            repulsive_vec = np.zeros(3)
            if avoid_obstacles:
                buffer = 0.12 
                for link in self.arm_links:
                    l_pos = link.get_pos().cpu().numpy()
                    for obs in self.obstacles:
                        o_min, o_max = obs.get_AABB()[0].cpu().numpy(), obs.get_AABB()[1].cpu().numpy()
                        closest_p = np.maximum(o_min, np.minimum(l_pos, o_max))
                        dist = np.linalg.norm(l_pos - closest_p)
                        
                        if dist < buffer:
                            w = (buffer - dist) / buffer 
                            push_dir = (l_pos - closest_p) / (dist + 1e-6)
                            if push_dir[2] < 0: 
                                push_dir[2] = 0
                            repulsive_vec += push_dir * 0.25 * w

            # --- 3. 计算合力 ---
            move_dir = attraction_vec + repulsive_vec
            if np.linalg.norm(move_dir) > speed:
                move_dir = move_dir / np.linalg.norm(move_dir) * speed

            # --- 4. 执行控制 ---
            q_goal = self.robot.inverse_kinematics(link=self.ee_link, pos=curr_ee_pos + move_dir, quat=downward_quat)
            
            if q_goal is not None:
                current_action = q_goal[:7] 
                # 记录当前的 action (转为 cpu numpy 以防 collector 报错)
                if hasattr(current_action, 'cpu'):
                    current_action = current_action.cpu().numpy()
                
                self.robot.control_dofs_position(current_action, self.motors_dof)
            
                # 保持夹爪状态逻辑 (根据你原有的逻辑补偿)
                if force == 0:
                    self.robot.control_dofs_position(np.array([finger_pos, finger_pos]), self.fingers_dof)
                else:
                    self.robot.control_dofs_force(np.array([force, force]), self.fingers_dof)

                # 数据收集钩子
                if callback is not None:
                    callback(action=current_action)
            
            self.scene.step()