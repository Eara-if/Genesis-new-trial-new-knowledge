import genesis as gs
import numpy as np
from array_gripper import VacuumGripper

class WaypointMultiPickTask:
    def __init__(self, scene, robot, target_list):
        self.scene = scene
        self.robot = robot
        self.target_list = target_list
        
        self.current_idx = 0
        self.SAFE_Z = 0.45  
        self.RELEASE_POS = np.array([0.2, -0.45, 0.45], dtype=np.float64) 
        self.OBSERVE_POS = np.array([0.1, 0.45, 0.55], dtype=np.float64) 
        
        self.ee_link = self.robot.links[7]
        self.gripper = VacuumGripper(self.scene, self.robot, 7, max_force=25.0)
        
        self.phase = 'return_to_observe' 
        self.active_target = None 
        self.current_goal = None
        self.recorded_goal = None 
        self.top_quat = np.array([0, 1, 0, 0], dtype=np.float64) 
        self.is_all_completed = False
        self.phase_timer = 0
        self.pause_timer = 0

    def start(self):
        self.gripper.setup_visual_attachment()
        self.current_goal = np.array(self.gripper.get_position(), dtype=np.float64)
        self.recorded_goal = self.current_goal.copy()

    def update_targets(self, new_targets):
        self.target_list = new_targets
        if self.phase == 'searching' and len(self.target_list) == 0:
            print("🎉 所有视野内的目标已清理完毕！")
            self.is_all_completed = True
        if self.phase == 'searching' and len(self.target_list) > 0:
            self.active_target = self.target_list[0]['pos']
            self._switch_phase('approach_top')

    # 在 WaypointMultiPickTask 类中新增奖励计算方法
    def compute_rewards(self):
        """计算双重奖励机制"""
        dense_reward = 0.0
        sparse_reward = 0.0

        # ========== 1. 稀疏奖励 (Sparse Reward) ==========
        # 只有当所有箱子清理完毕，或者成功完成单次抓取并释放时给予奖励
        if self.is_all_completed:
            sparse_reward = 10.0  # 终极任务完成奖励
        elif self.phase == 'release' and self.phase_timer == 1:
            sparse_reward = 1.0   # 单个目标成功抓取并放置奖励

        # ========== 2. 密集奖励 (Dense Reward) ==========
        ee_pos = np.array(self.gripper.get_position(), dtype=np.float64)
        
        if self.active_target is not None:
            # a. 接近奖励 (Distance Reward)：距离目标越近，奖励越高
            # 使用高斯衰减函数 np.exp(-k * dist)，使奖励平滑过渡
            dist_to_target = np.linalg.norm(ee_pos[:3] - self.active_target[:3])
            approach_reward = np.exp(-5.0 * dist_to_target) 
            
            # b. 吸附状态奖励 (Attachment Reward)：成功吸附给予持续的阶段性奖励
            attach_reward = 0.0
            if self.gripper.is_attached:
                attach_reward = 2.0
                
                # c. 抬升奖励 (Lifting Reward)：吸住并成功抬起的高度越高，奖励越大
                lift_height = max(0.0, ee_pos[2] - self.active_target[2])
                lift_reward = min(1.0, lift_height / 0.2) * 2.0  # 假设最高抬起 0.2m
                attach_reward += lift_reward

            dense_reward = approach_reward + attach_reward

        # 惩罚项：如果发生空中脱落，给予负向密集惩罚
        if self.phase == 'abort':
            dense_reward = -2.0

        return float(dense_reward), float(sparse_reward)

    # 修改原有的 get_expert_data 方法，把奖励抛出
    def get_expert_data(self):
        q = self.robot.get_dofs_position().detach().cpu().numpy()[:6]
        qd = self.robot.get_dofs_velocity().detach().cpu().numpy()[:6]
        delta_pos = self.current_goal - self.recorded_goal if self.recorded_goal is not None else np.zeros(3)
        self.recorded_goal = self.current_goal.copy()
        
        tactile_data = self.gripper.get_tactile_data()
        
        # [新增] 计算当前步的双重奖励
        dense_r, sparse_r = self.compute_rewards()
        
        return {
            "joint_pos": q,
            "joint_vel": qd,
            "delta_pos": delta_pos,
            "gripper_attached": self.gripper.is_attached,
            "tactile": tactile_data,
            "reward_dense": dense_r,   # 抛出密集奖励
            "reward_sparse": sparse_r  # 抛出稀疏奖励
        }

    def step(self):
        if self.is_all_completed: return True
        
        is_detached = self.gripper.check_detachment()
        if is_detached and self.phase in ['lift_to_safe', 'transport_safe', 'stabilize']:
            print("🚨 [系统异常] 检测到物体在空中脱落！")
            self.active_target = None
            self._switch_phase('abort')
            
        if self.active_target is None and self.phase not in ['release', 'return_to_observe', 'searching', 'abort']: 
            return False
        
        ee_pos = np.array(self.gripper.get_position(), dtype=np.float64)
        max_step = 0.007 
        self.phase_timer += 1
        goal_dest = self.current_goal.copy()

        if self.phase == 'return_to_observe':
            goal_dest = self.OBSERVE_POS
            if np.linalg.norm(goal_dest[:2] - ee_pos[:2]) < 0.03:
                if self.phase_timer > 30: self._switch_phase('searching')
        elif self.phase == 'searching':
            goal_dest = self.OBSERVE_POS
        elif self.phase == 'approach_top':
            goal_dest = np.array([self.active_target[0], self.active_target[1], self.SAFE_Z])
            if np.linalg.norm(goal_dest[:2] - ee_pos[:2]) < 0.01: self._switch_phase('descend')
        elif self.phase == 'descend':
            goal_dest = self.active_target
            if ee_pos[2] < self.active_target[2] + 0.01:
                if self.gripper.activate_suction_by_pos(self.active_target):
                    self._switch_phase('lift_to_safe')
                elif self.phase_timer > 100: self._switch_phase('return_to_observe')
        elif self.phase == 'lift_to_safe':
            goal_dest = np.array([ee_pos[0], ee_pos[1], self.SAFE_Z])
            if ee_pos[2] >= (self.SAFE_Z - 0.02): self._switch_phase('transport_safe')
        elif self.phase == 'transport_safe':
            goal_dest = self.RELEASE_POS
            if np.linalg.norm(goal_dest[:2] - ee_pos[:2]) < 0.02: 
                self._switch_phase('stabilize')
                self.pause_timer = 40
        elif self.phase == 'stabilize':
            goal_dest = self.RELEASE_POS
            self.pause_timer -= 1
            if self.pause_timer <= 0: self._switch_phase('release')
        elif self.phase == 'release':
            self.gripper.deactivate_suction()
            if self.phase_timer > 30:
                self.active_target = None
                self._switch_phase('return_to_observe') 
        elif self.phase == 'abort':
            goal_dest = np.array([ee_pos[0], ee_pos[1], self.SAFE_Z])
            if ee_pos[2] >= (self.SAFE_Z - 0.02): self._switch_phase('return_to_observe')

        move_vec = goal_dest - self.current_goal
        dist = np.linalg.norm(move_vec)
        if dist > max_step: move_vec = (move_vec / dist) * max_step
        self.current_goal += move_vec
        
        q_target = self.robot.inverse_kinematics(link=self.ee_link, pos=self.current_goal, quat=self.top_quat)
        if q_target is not None:
            self.robot.control_dofs_position(q_target[:6], np.arange(6))
        
        return self.is_all_completed

    def _switch_phase(self, new_phase):
        self.phase = new_phase
        self.phase_timer = 0