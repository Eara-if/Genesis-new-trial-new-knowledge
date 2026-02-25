import torch
import numpy as np
import genesis as gs

class VacuumGripper:
    def __init__(self, scene, robot, suction_cups, suction_link_idx, max_force=25.0):
        self.scene = scene
        self.robot = robot
        self.suction_cups = suction_cups
        self.suction_link_id = suction_link_idx
        self.is_attached = False
        self.attached_link_idx = None
        self.attached_obj = None  
        self.max_suction_force = max_force # 最大吸力（法向阈值）
        self.current_mu = 0.5 
        self.last_ee_vel = None
        self.dt = 0.01
        self.lift_timer = 0
        self.step_count = 0
        self.smooth_accel = torch.zeros(3, device=torch.device("cpu")) 

    def _rotate_vector_by_quat(self, v_local, q):
        # 向量旋转逻辑，确保在计算偏移时坐标系正确
        w, x, y, z = q[0], q[1], q[2], q[3]
        vx, vy, vz = v_local[0], v_local[1], v_local[2]
        res_x = vx * (1 - 2*y**2 - 2*z**2) + vy * (2*x*y - 2*w*z) + vz * (2*x*z + 2*w*y)
        res_y = vx * (2*x*y + 2*w*z) + vy * (1 - 2*x**2 - 2*z**2) + vz * (2*y*z - 2*w*x)
        res_z = vx * (2*x*z - 2*w*y) + vy * (2*y*z + 2*w*x) + vz * (1 - 2*x**2 - 2*y**2)
        return torch.stack([res_x, res_y, res_z])

    def setup_visual_attachment(self):
        # 初始化吸盘视觉位置，并建立永久焊死约束
        ee_link = self.robot.links[self.suction_link_id]
        ee_pos, ee_quat = ee_link.get_pos(), ee_link.get_quat()
        offsets = [
            torch.tensor([0.02, 0.02, 0.015], device=ee_pos.device),
            torch.tensor([-0.02, 0.02, 0.015], device=ee_pos.device),
            torch.tensor([0.02, -0.02, 0.015], device=ee_pos.device),
            torch.tensor([-0.02, -0.02, 0.015], device=ee_pos.device)
        ]
        for i, cup in enumerate(self.suction_cups):
            world_offset = self._rotate_vector_by_quat(offsets[i], ee_quat)
            cup.set_pos(ee_pos + world_offset)
            cup.set_quat(ee_quat)
            self.scene.sim.rigid_solver.add_weld_constraint(cup.links[0].idx, ee_link.idx)

    def activate_suction(self, target_entity):
        if self.is_attached: return True
        aabb = target_entity.get_AABB()
        obj_max_z = aabb[1, 2]
        for i, cup in enumerate(self.suction_cups):
            cup_pos = cup.get_pos()
            surface_dist = cup_pos[2] - (cup.morph.height / 2) - obj_max_z
            # 距离检测：吸盘靠近物体表面时激活
            if -0.012 <= surface_dist <= 0.008:
                try:
                    target_idx = target_entity.links[0].idx
                    self.attached_link_idx = target_idx
                    self.attached_obj = target_entity 
                    
                    # 动态读取来自 assets_manager 的摩擦力属性
                    if hasattr(target_entity, 'friction'):
                        self.current_mu = target_entity.friction
                    
                    # 建立临时物理连接
                    self.scene.sim.rigid_solver.add_weld_constraint(self.attached_link_idx, self.suction_link_id)
                    self.is_attached = True
                    self.lift_timer = 25 # 初始稳定时间
                    print(f"🧲 [吸附激活] 物体已锁定 | 材质摩擦系数 μ={self.current_mu:.2f}")
                    return True
                except Exception as e:
                    print(f"⚠️ [激活失败] 发生异常: {e}")
                    return False
        return False

    def check_detachment(self):
        """核心功能：计算加速度及受力，判定脱附状态"""
        if not self.is_attached or self.attached_obj is None: return False
        
        self.step_count += 1
        if self.lift_timer > 0:
            self.lift_timer -= 1
            return False
            
        # 1. 物理参数准备
        base_mass = 0.5000 # 基准质量
        current_scale = self.attached_obj.morph.scale
        dynamic_mass = base_mass * (current_scale ** 3) 
        g = 9.81

        ee_link = self.robot.links[self.suction_link_id]
        curr_vel_raw = ee_link.get_vel()[:3]
        
        # 强制同步到 CPU 处理，防止 Segfault
        curr_vel = curr_vel_raw.detach().cpu()
        
        if self.last_ee_vel is None:
            self.last_ee_vel = curr_vel
            return False
        
        # 2. 加速度计算与平滑滤波
        raw_accel = (curr_vel - self.last_ee_vel) / self.dt
        raw_accel = torch.clamp(raw_accel, -20.0, 20.0) 
        self.smooth_accel = 0.85 * self.smooth_accel + 0.15 * raw_accel 
        self.last_ee_vel = curr_vel
        
        ax, ay, az = self.smooth_accel[0].item(), self.smooth_accel[1].item(), self.smooth_accel[2].item()

        # 3. 受力逻辑计算
        # 垂直脱离力（重力 + 垂直惯性力）
        pulling_force_z = dynamic_mass * max(0.0, g + az)
        
        # 实际接触正压力（Pressure）：由吸力抵消掉脱离力后的剩余压力
        # 当 pulling_force_z 接近 max_suction_force 时，正压力趋近于 0，摩擦力也将消失
        actual_pressure = max(0.0, self.max_suction_force - pulling_force_z)
        
        # 水平惯性剪切力 (F = m * sqrt(ax^2 + ay^2))
        applied_shear_force = dynamic_mass * torch.norm(self.smooth_accel[:2]).item()
        
        # 最大静摩擦力阈值 (f_max = μ * N)
        friction_threshold = actual_pressure * self.current_mu
        
        # 4. 终端实时监控打印
        if self.step_count % 40 == 0:
            print("-" * 50)
            print(f"📈 [传感器数据] 步数: {self.step_count}")
            print(f"   质量 (kg): {dynamic_mass:.2f}")
            print(f"   加速度 (m/s²): X: {ax:6.2f} | Y: {ay:6.2f} | Z: {az:6.2f}")
            print(f"   法向力 (N): 垂直拉力: {pulling_force_z:.2f} | 吸力上限: {self.max_suction_force:.2f}")
            print(f"   接触压力 (N): 实际压力: {actual_pressure:.2f} | 压力阈值: {self.max_suction_force:.2f}")
            print(f"   摩擦力 (N): 剪切惯性力: {applied_shear_force:.2f} | 静摩擦阈值: {friction_threshold:.2f} (μ={self.current_mu:.2f})")

        # 5. 脱附判定逻辑
        # 情况 A: 垂直加速度过大导致“拉断”
        if pulling_force_z > self.max_suction_force:
            print(f"❌ [脱附告警] 垂直载荷过载！物体掉落。")
            self.deactivate_suction()
            return True
        
        # 情况 B: 水平加速度过大导致“打滑”
        if applied_shear_force > friction_threshold:
            print(f"❌ [脱附告警] 水平加速度过快，摩擦失效导致打滑！")
            self.deactivate_suction()
            return True
            
        return False

    def deactivate_suction(self):
        if self.is_attached:
            try:
                # 安全删除约束
                if self.attached_link_idx is not None:
                    self.scene.sim.rigid_solver.delete_weld_constraint(self.attached_link_idx, self.suction_link_id)
            except Exception:
                pass 
            self.is_attached = False
            self.attached_link_idx = None
            self.attached_obj = None
            self.last_ee_vel = None
            self.smooth_accel = torch.zeros(3)
            print("🔓 [吸附解除] 约束已安全断开")

    def get_position(self):
        # 获取末端位置，同样强制 CPU 同步
        pos = self.robot.links[self.suction_link_id].get_pos()
        return pos.detach().cpu().numpy().astype(np.float64)