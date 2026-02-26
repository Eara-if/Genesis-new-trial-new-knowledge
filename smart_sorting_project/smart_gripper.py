import numpy as np

class SmartParallelGripper:
    def __init__(self, robot, fingers_dof, dt=0.005):
        self.robot = robot
        self.fingers_dof = fingers_dof
        self.dt = dt
        self.last_vel = np.zeros(3)
        self.g = 9.81
        
    def compute_required_force(self, obj_mass, obj_mu, current_accel_z=0.0):
        safe_mu = max(obj_mu, 0.1)
        total_g_load = self.g + max(current_accel_z, 0)
        min_force = (obj_mass * total_g_load) / (2 * safe_mu)
        target_force = min_force * 1.5 
        return max(target_force, 5.0)

    def print_status(self, step, mass, mu, applied_force, required_force, velocity):
        curr_vel = velocity
        accel = (curr_vel - self.last_vel) / self.dt
        self.last_vel = curr_vel
        friction_force_provided = 2 * mu * applied_force
        load_force = mass * (self.g + accel[2])
        status = "✅ 锁定" if friction_force_provided > load_force else "⚠️ 打滑风险"

        print("-" * 60)
        print(f"📊 [智能夹爪监测] 步数: {step}")
        print(f"   📦 物体属性: 质量 m={mass:.2f}kg | 摩擦系数 μ={mu:.2f}")
        print(f"   🚀 运动监控: 速度 v={np.linalg.norm(curr_vel):.2f}m/s | 加速度 a_z={accel[2]:.2f}m/s²")
        print(f"   🦾 力控状态: 当前施力={applied_force:.2f}N | 理论阈值={required_force:.2f}N")
        print(f"   ⚖️ 摩擦判定: {status} (最大静摩擦={friction_force_provided:.2f}N vs 负载={load_force:.2f}N)")