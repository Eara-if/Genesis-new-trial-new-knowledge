import numpy as np


class TactileGenerator:
    """基于吸盘受力状态生成近似二维触觉热图。"""

    def __init__(self, grid_size=(16, 16), max_force=25.0):
        self.grid_h, self.grid_w = grid_size
        self.max_force = float(max_force)
        self._last_tactile = self._empty_state()

    def _empty_state(self):
        zeros = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        return {
            "pressure_map": zeros.copy(),
            "shear_map": zeros.copy(),
            "contact_mask": zeros.copy(),
            "meta": {
                "is_attached": False,
                "normal_force": 0.0,
                "shear_force": 0.0,
                "friction_mu": 0.0,
                "slip_ratio": 0.0,
            },
        }

    def reset(self):
        self._last_tactile = self._empty_state()
        return self._last_tactile

    def _gaussian_blob(self, center_x, center_y, sigma_x, sigma_y):
        yy, xx = np.mgrid[0:self.grid_h, 0:self.grid_w]
        return np.exp(
            -(
                ((xx - center_x) ** 2) / (2.0 * sigma_x ** 2)
                + ((yy - center_y) ** 2) / (2.0 * sigma_y ** 2)
            )
        )

    def update(self, is_attached, normal_force, shear_force, friction_mu, accel_xy=None):
        if not is_attached:
            return self.reset()

        normal_force = float(max(0.0, normal_force))
        shear_force = float(max(0.0, shear_force))
        friction_mu = float(max(0.0, friction_mu))
        accel_xy = np.asarray(accel_xy if accel_xy is not None else [0.0, 0.0], dtype=np.float32)

        # 以受力方向轻微偏移热图中心，模拟滑移趋势
        shift_x = float(np.clip(accel_xy[0], -1.0, 1.0)) * 2.0
        shift_y = float(np.clip(accel_xy[1], -1.0, 1.0)) * 2.0
        cx = (self.grid_w - 1) / 2.0 + shift_x
        cy = (self.grid_h - 1) / 2.0 + shift_y

        normal_ratio = np.clip(normal_force / max(self.max_force, 1e-6), 0.0, 1.0)
        shear_capacity = max(normal_force * max(friction_mu, 1e-6), 1e-6)
        slip_ratio = np.clip(shear_force / shear_capacity, 0.0, 2.0)

        sigma_scale = 1.0 + 1.8 * (1.0 - normal_ratio)
        pressure_blob = self._gaussian_blob(cx, cy, 2.2 * sigma_scale, 2.2 * sigma_scale)
        pressure_map = pressure_blob * normal_ratio
        pressure_map = pressure_map.astype(np.float32)

        if pressure_map.max() > 0:
            pressure_map /= pressure_map.max()
            pressure_map *= normal_ratio

        # 剪切场沿运动方向拉伸，并在边缘更明显
        dx = accel_xy[0]
        dy = accel_xy[1]
        direction_norm = float(np.linalg.norm([dx, dy]))
        if direction_norm < 1e-6:
            dir_x, dir_y = 1.0, 0.0
        else:
            dir_x, dir_y = dx / direction_norm, dy / direction_norm

        yy, xx = np.mgrid[0:self.grid_h, 0:self.grid_w]
        rel_x = xx - cx
        rel_y = yy - cy
        directional_term = np.abs(rel_x * dir_x + rel_y * dir_y)
        directional_term = directional_term / (directional_term.max() + 1e-6)

        shear_map = directional_term * pressure_blob * np.clip(slip_ratio, 0.0, 1.5)
        shear_map = shear_map.astype(np.float32)

        if shear_map.max() > 0:
            shear_map /= shear_map.max()
            shear_map *= min(1.0, slip_ratio)

        contact_mask = (pressure_map > max(0.08, 0.25 * normal_ratio)).astype(np.float32)

        self._last_tactile = {
            "pressure_map": pressure_map,
            "shear_map": shear_map,
            "contact_mask": contact_mask,
            "meta": {
                "is_attached": True,
                "normal_force": normal_force,
                "shear_force": shear_force,
                "friction_mu": friction_mu,
                "slip_ratio": float(slip_ratio),
            },
        }
        return self._last_tactile

    def get_last_tactile(self):
        return self._last_tactile