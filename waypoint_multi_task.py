import genesis as gs
import numpy as np
from array_gripper import VacuumGripper

class WaypointMultiPickTask:
    def __init__(self, scene, robot, suction_cups, obj_list):
        self.scene = scene
        self.robot = robot
        self.suction_cups = suction_cups
        self.obj_list = obj_list
        self.current_obj_idx = 0
        
        self.dt = 0.01
        self.SAFE_Z = 0.45  
        self.RELEASE_POS = np.array([0.3, -0.4, 0.45], dtype=np.float64) 
        
        # 初始化吸盘，确保最大力矩足够大
        self.gripper = VacuumGripper(self.scene, self.robot, self.suction_cups, 7, max_force=25.0)
        self.gripper.dt = self.dt
        
        self.phase = 'approach_top' 
        self.current_goal = None
        self.lock_pick_xy = None  
        self.top_quat = np.array([0, 1, 0, 0], dtype=np.float64) 
        self.is_all_completed = False
        
        # 核心：用于控制停留时间和防止卡死的计时器
        self.phase_timer = 0 

    def _get_current_obj(self):
        return self.obj_list[self.current_obj_idx]

    def _get_obj_surface_z(self, obj):
        aabb = obj.get_AABB()
        # 确保 AABB 数据从 GPU 同步到 CPU，防止跨设备计算错误
        return aabb[1, 2].cpu().item() - 0.002

    def start(self):
        self.gripper.setup_visual_attachment()
        # 初始目标点设为当前实际位置，防止第一帧产生巨大的位移冲力
        self.current_goal = np.array(self.gripper.get_position(), dtype=np.float64)
        print(f"🚀 任务启动 | 目标总数: {len(self.obj_list)}")

    def step(self):
        if self.is_all_completed: return True
        
        self.gripper.check_detachment()
        ee_pos = np.array(self.gripper.get_position(), dtype=np.float64)
        target_obj = self._get_current_obj()
        obj_pos_realtime = target_obj.get_pos().cpu().numpy().astype(np.float64)
        
        max_step = 0.004 # 略微加快移动速度
        self.phase_timer += 1
        
        # 默认目标点初始化
        goal_dest = self.current_goal.copy()

        # --- 状态机逻辑 ---
        
        if self.phase == 'approach_top':
            goal_dest = np.array([obj_pos_realtime[0], obj_pos_realtime[1], self.SAFE_Z])
            if np.linalg.norm(goal_dest - ee_pos) < 0.02:
                self._switch_phase('descend')

        elif self.phase == 'descend':
            surface_z = self._get_obj_surface_z(target_obj)
            goal_dest = np.array([obj_pos_realtime[0], obj_pos_realtime[1], surface_z])
            # 激活吸盘
            if self.gripper.activate_suction(target_obj):
                self.lock_pick_xy = obj_pos_realtime[:2].copy()
                self._switch_phase('lift_to_safe')
            elif ee_pos[2] <= surface_z + 0.001:
                self.lock_pick_xy = obj_pos_realtime[:2].copy()
                self._switch_phase('lift_to_safe')

        elif self.phase == 'lift_to_safe':
            goal_dest = np.array([self.lock_pick_xy[0], self.lock_pick_xy[1], self.SAFE_Z])
            # 如果到达高度或因物理碰撞卡住（超时），进入平移阶段
            if ee_pos[2] >= (self.SAFE_Z - 0.015) or self.phase_timer > 150:
                self._switch_phase('transport_safe')

        elif self.phase == 'transport_safe':
            goal_dest = self.RELEASE_POS
            if np.linalg.norm(goal_dest[:2] - ee_pos[:2]) < 0.02:
                self._switch_phase('wait_for_stop')

        elif self.phase == 'wait_for_stop':
            # 目标：释放前强制停止。停留 30 步（约 0.3s）足以让机械臂动量消失
            goal_dest = self.RELEASE_POS
            if self.phase_timer > 30: 
                self._switch_phase('release')

        elif self.phase == 'release':
            # 1. 立即释放
            self.gripper.deactivate_suction()
            
            # 2. 原地停留 20 步，确保吸盘物理上脱离物体，然后再上抬
            if self.phase_timer < 20:
                goal_dest = self.RELEASE_POS
            else:
                goal_dest = self.RELEASE_POS + np.array([0, 0, 0.1])
            
            # 3. 完成判断
            if self.phase_timer > 50 and np.linalg.norm(goal_dest - ee_pos) < 0.03:
                if self.current_obj_idx < len(self.obj_list) - 1:
                    print(f"✅ 盒子 {self.current_obj_idx + 1} 完成")
                    self.current_obj_idx += 1
                    self._switch_phase('approach_top')
                else:
                    print(f"🎉 任务全部完成 | ID: 10245102480")
                    self.is_all_completed = True

        # --- 运动平滑处理 (防止 Segfault 的关键) ---
        move_vec = goal_dest - self.current_goal
        dist = np.linalg.norm(move_vec)
        if dist > max_step:
            move_vec = (move_vec / dist) * max_step
        self.current_goal += move_vec
        
        # IK 求解：link=7 对应 UR5e 的末端
        q_target = self.robot.inverse_kinematics(
            link=self.robot.links[7], 
            pos=self.current_goal, 
            quat=self.top_quat
        )
        
        # 增加有效性检查，防止 None 导致的控制异常
        if q_target is not None:
            # 仅控制 UR5e 的 6 个关节，避免 index out of range
            target_angles = q_target[:6]
            self.robot.control_dofs_position(target_angles, np.arange(6))
        
        return self.is_all_completed

    def _switch_phase(self, new_phase):
        """核心辅助：重置计时器并同步目标位置"""
        self.phase = new_phase
        self.phase_timer = 0
        # 切换状态时，让目标点重新对齐实际位置，消除累积误差
        self.current_goal = np.array(self.gripper.get_position(), dtype=np.float64)