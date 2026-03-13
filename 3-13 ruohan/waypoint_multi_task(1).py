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
        self.top_quat = np.array([0, 1, 0, 0], dtype=np.float64) 
        self.is_all_completed = False
        self.phase_timer = 0
        
        # 新增：用于静定缓冲的计时器
        self.pause_timer = 0

    def start(self):
        self.gripper.setup_visual_attachment()
        self.current_goal = np.array(self.gripper.get_position(), dtype=np.float64)

    def update_targets(self, new_targets):
        self.target_list = new_targets
        
        if self.phase == 'searching' and len(self.target_list) == 0:
            print("🎉 所有视野内的目标已清理完毕！")
            
        if self.phase == 'searching' and len(self.target_list) > 0:
            self.active_target = self.target_list[0]['pos']
            self._switch_phase('approach_top')

    def step(self):
        if self.is_all_completed: return True
        
        # 【核心架构升级】：多任务闭环的异常捕获机制
        # 始终更新末端加速度缓冲，并在承重阶段严格捕获脱附返回值
        is_detached = self.gripper.check_detachment()
        if is_detached and self.phase in ['lift_to_safe', 'transport_safe', 'stabilize']:
            print("🚨 [系统异常] 检测到物体在空中脱落！立即中止当前搬运，启动恢复程序。")
            self.active_target = None # 彻底清空内存中失效的目标坐标
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
                if self.phase_timer > 30: 
                    self._switch_phase('searching')

        elif self.phase == 'searching':
            goal_dest = self.OBSERVE_POS

        elif self.phase == 'approach_top':
            goal_dest = np.array([self.active_target[0], self.active_target[1], self.SAFE_Z])
            if np.linalg.norm(goal_dest[:2] - ee_pos[:2]) < 0.01: 
                self._switch_phase('descend')

        elif self.phase == 'descend':
            goal_dest = self.active_target
            if ee_pos[2] < self.active_target[2] + 0.01:
                if self.gripper.activate_suction_by_pos(self.active_target):
                    self._switch_phase('lift_to_safe')
                elif self.phase_timer > 100:
                    self._switch_phase('return_to_observe')

        elif self.phase == 'lift_to_safe':
            goal_dest = np.array([ee_pos[0], ee_pos[1], self.SAFE_Z])
            if ee_pos[2] >= (self.SAFE_Z - 0.02): 
                self._switch_phase('transport_safe')

        elif self.phase == 'transport_safe':
            goal_dest = self.RELEASE_POS
            # 【核心功能植入】：到达释放区后，不立刻撒手，切入静定缓冲
            if np.linalg.norm(goal_dest[:2] - ee_pos[:2]) < 0.02: 
                self._switch_phase('stabilize')
                self.pause_timer = 40 # 设定 0.4 秒的动能耗散时间
                
        elif self.phase == 'stabilize':
            # 保持位姿不动，等待物理引擎中因为急停带来的惯性晃动平息
            goal_dest = self.RELEASE_POS
            self.pause_timer -= 1
            if self.pause_timer <= 0:
                self._switch_phase('release')

        elif self.phase == 'release':
            self.gripper.deactivate_suction()
            if self.phase_timer > 30:
                self.active_target = None
                self._switch_phase('return_to_observe') 
                
        elif self.phase == 'abort':
            # 【工业级容错策略】：脱落后先拉升至安全高度，避免碰倒其他堆叠塔
            goal_dest = np.array([ee_pos[0], ee_pos[1], self.SAFE_Z])
            if ee_pos[2] >= (self.SAFE_Z - 0.02): 
                # 抬升完毕后，强制系统退回观察位，视觉重新扫描桌面寻找掉落的箱子
                self._switch_phase('return_to_observe')

        # 运动学反解与下发
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