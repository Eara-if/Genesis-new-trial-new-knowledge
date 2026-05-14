import torch
import numpy as np
import genesis as gs
from tactile_generator import TactileGenerator


class VacuumGripper:
    def __init__(self, scene, robot, suction_link_idx, max_force=25.0):
        self.scene = scene
        self.robot = robot
        self.suction_link_id = suction_link_idx
        self.is_attached = False
        self.attached_link_idx = None
        self.attached_obj = None
        self.max_suction_force = max_force
        self.current_mu = 0.5
        self.last_ee_vel = None
        self.dt = 0.01
        self.lift_timer = 0
        self.step_count = 0
        self.smooth_accel = torch.zeros(3, device=torch.device("cpu"))
        self.last_normal_force = 0.0
        self.last_shear_force = 0.0

        self.tactile_generator = TactileGenerator(
            grid_size=(16, 16),
            max_force=max_force
        )

    def setup_visual_attachment(self):
        # 取消了冗余实体的生成
        pass

    def activate_suction_by_pos(self, vision_pos, threshold=0.05):
        if self.is_attached:
            return True

        candidate_entities = []

        # 1. 收集所有在二维 XY 容差范围内的候选实体
        for entity in self.scene.entities:
            if entity == self.robot:
                continue

            obj_pos = entity.get_pos().cpu().numpy()
            dist_2d = np.linalg.norm(obj_pos[:2] - vision_pos[:2])

            if dist_2d < threshold:
                candidate_entities.append(entity)

        if not candidate_entities:
            return False

        # 2. 在候选列表中，挑出 Z 坐标最高的物体进行绑定
        target_entity = max(candidate_entities, key=lambda e: e.get_pos()[2].item())
        return self.activate_suction(target_entity)

    def activate_suction(self, target_entity):
        if self.is_attached:
            return True

        try:
            target_idx = target_entity.links[0].idx
            self.attached_link_idx = target_idx
            self.attached_obj = target_entity

            if hasattr(target_entity, 'friction'):
                self.current_mu = target_entity.friction

            self.scene.sim.rigid_solver.add_weld_constraint(
                self.attached_link_idx,
                self.suction_link_id
            )

            self.is_attached = True
            self.lift_timer = 25

            self.tactile_generator.update(
                True,
                0.15 * self.max_suction_force,
                0.0,
                self.current_mu,
                accel_xy=[0.0, 0.0]
            )

            print(f"🧲 [视觉引导吸附] 锁定目标物体 (μ={self.current_mu:.2f})")
            return True

        except Exception as e:
            print(f"⚠️ 激活失败: {e}")
            return False

    def check_detachment(self):
        if not self.is_attached or self.attached_obj is None:
            return False

        self.step_count += 1

        if self.lift_timer > 0:
            self.lift_timer -= 1
            return False

        base_mass = 0.5
        current_scale = self.attached_obj.morph.scale
        dynamic_mass = base_mass * (current_scale ** 3)
        g = 9.81

        ee_link = self.robot.links[self.suction_link_id]
        curr_vel = ee_link.get_vel()[:3].detach().cpu()

        if self.last_ee_vel is None:
            self.last_ee_vel = curr_vel
            return False

        raw_accel = (curr_vel - self.last_ee_vel) / self.dt
        raw_accel = torch.clamp(raw_accel, -20.0, 20.0)
        self.smooth_accel = 0.85 * self.smooth_accel + 0.15 * raw_accel
        self.last_ee_vel = curr_vel

        az = self.smooth_accel[2].item()
        pulling_force_z = dynamic_mass * max(0.0, g + az)
        actual_pressure = max(0.0, self.max_suction_force - pulling_force_z)
        applied_shear_force = dynamic_mass * torch.norm(self.smooth_accel[:2]).item()
        friction_threshold = actual_pressure * self.current_mu

        self.last_normal_force = float(actual_pressure)
        self.last_shear_force = float(applied_shear_force)

        self.tactile_generator.update(
            is_attached=True,
            normal_force=self.last_normal_force,
            shear_force=self.last_shear_force,
            friction_mu=self.current_mu,
            accel_xy=self.smooth_accel[:2].detach().cpu().numpy(),
        )

        if self.step_count % 50 == 0:
            print("-" * 50)
            print(f"📊 状态: 质量={dynamic_mass:.3f}kg | az={az:6.2f} | 净拉力={pulling_force_z:.2f}N")
            print(f"📊 摩擦锥: 剪切力={applied_shear_force:.2f}N (滑移阈值: {friction_threshold:.2f}N | μ={self.current_mu:.2f})")

        # 脱附判定：法向拉断 或 切向滑移
        if pulling_force_z > self.max_suction_force:
            print(f"⚠️ [法向脱附] 垂直过载: {pulling_force_z:.2f}N (吸力上限: {self.max_suction_force:.2f}N)")
            self.deactivate_suction()
            return True

        if applied_shear_force > friction_threshold:
            print(f"⚠️ [切向滑移] 剪切突破: {applied_shear_force:.2f}N (当前静摩擦上限: {friction_threshold:.2f}N)")
            self.deactivate_suction()
            return True

        return False

    def deactivate_suction(self):
        if self.is_attached:
            try:
                if self.attached_link_idx is not None:
                    self.scene.sim.rigid_solver.delete_weld_constraint(
                        self.attached_link_idx,
                        self.suction_link_id
                    )
            except Exception:
                pass

            self.is_attached = False
            self.attached_link_idx = None
            self.attached_obj = None
            self.last_normal_force = 0.0
            self.last_shear_force = 0.0
            self.tactile_generator.reset()
            print("🔓 吸附解除")

    def get_position(self):
        pos = self.robot.links[self.suction_link_id].get_pos()
        return pos.detach().cpu().numpy().astype(np.float64)

    def get_tactile_data(self):
        return self.tactile_generator.get_last_tactile()