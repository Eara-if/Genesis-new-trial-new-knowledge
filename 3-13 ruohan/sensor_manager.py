import numpy as np
import cv2

class SensorManager:
    """
    统一管理多相机与数据采集。
    
    修复说明：
    1. 架构级重构：彻底放弃应用层对底层物理连杆 (link/geom) ID 的猜测。
    2. 配合全局的 segmentation_level='entity'，现在只需映射 entity.idx + 1。
    """

    PALETTE = {
        0: (40, 40, 40),      # background/Sky: 深灰
        1: (120, 120, 120),   # floor: 灰色
        2: (0, 255, 0),       # box: 绿色 (BGR)
        3: (0, 0, 255),       # robot: 红色 (BGR)
    }

    def __init__(self, scene, robot, obj_list, image_size=(640, 480)):
        self.scene = scene
        self.robot = robot
        self.obj_list = obj_list
        self.image_size = image_size

        self.cameras = {}
        self.camera_specs = {}

        self._mappings_built = False
        self.sem_lut = None
        self.inst_lut = None

        self._add_default_cameras()

    def _build_mappings(self):
        """
        构建基于实体 (Entity) 级别的查找表。
        """
        semantic_mapping = {0: 0} 
        instance_mapping = {0: 0}

        for entity in self.scene.entities:
            # 确定语义分类
            if entity == self.robot:
                sem_id, inst_id = 3, 3  # 机器人
            elif any(entity == obj for obj in self.obj_list):
                sem_id = 2             # 盒子
                try:
                    e_idx = int(entity.idx)
                    inst_id = 10 + (e_idx - 2)
                except:
                    inst_id = 10
            elif int(entity.idx) == 0:
                sem_id, inst_id = 1, 1  # 地面
            else:
                sem_id, inst_id = 1, 1

            # 🛑 核心修复点 2：直接使用 entity 层级的偏移
            # 在 entity 分割模式下，0 通常预留给背景天空，实体的像素值为 entity.idx + 1
            render_id = int(entity.idx) + 1
            semantic_mapping[render_id] = sem_id
            instance_mapping[render_id] = inst_id

        # 构建 Numpy 查找表
        max_id = max(semantic_mapping.keys()) if semantic_mapping else 0
        lut_size = max(max_id + 1, 1024)
        
        self.sem_lut = np.zeros(lut_size, dtype=np.int32)
        self.inst_lut = np.zeros(lut_size, dtype=np.int32)

        for k, v in semantic_mapping.items():
            if k < lut_size:
                self.sem_lut[k] = v
        for k, v in instance_mapping.items():
            if k < lut_size:
                self.inst_lut[k] = v

    def _apply_lut(self, raw_mask, lut):
        if raw_mask is None: return None
        raw_mask = np.asarray(raw_mask, dtype=np.int32)
        safe_mask = np.where((raw_mask >= 0) & (raw_mask < len(lut)), raw_mask, 0)
        return lut[safe_mask]

    def get_object_states(self):
        states = []
        for i, obj in enumerate(self.obj_list):
            try:
                pos = obj.get_pos().detach().cpu().numpy().tolist()
            except:
                pos = [0.0, 0.0, 0.0]
            states.append({"id": i, "name": f"box_{i}", "pos": pos})
        return states

    def _add_default_cameras(self):
        self.add_camera("top_cam", self.image_size, (0.5, 0.0, 1.2), (0.5, 0.0, 0.0), 50)
        self.add_camera("front_cam", self.image_size, (0.1, -0.65, 0.55), (0.55, 0.0, 0.1), 60)
        self.add_camera("wrist_cam", self.image_size, (0.3, 0.0, 0.45), (0.55, 0.0, 0.05), 65, True)

    def add_camera(self, name, res, pos, lookat, fov, is_dynamic=False):
        cam = self.scene.add_camera(res=res, pos=pos, lookat=lookat, fov=fov)
        self.cameras[name] = cam
        self.camera_specs[name] = {"res": tuple(res), "pos": tuple(pos), "lookat": tuple(lookat), "fov": float(fov), "is_dynamic": bool(is_dynamic)}
        return cam

    def update_dynamic_cameras(self):
        spec = self.camera_specs.get("wrist_cam")
        cam = self.cameras.get("wrist_cam")
        if not cam or not spec: return
        try:
            ee_link = self.robot.links[7]
            pos = ee_link.get_pos().detach().cpu().numpy()
            cam_pos = pos + np.array([-0.05, 0.0, 0.06])
            lookat = pos + np.array([0.2, 0.0, -0.1])
            spec["pos"], spec["lookat"] = tuple(cam_pos), tuple(lookat)
            if hasattr(cam, "set_pose"):
                cam.set_pose(pos=spec["pos"], lookat=spec["lookat"])
        except: pass

    def get_intrinsics(self, name):
        s = self.camera_specs[name]
        f = (s["res"][1] / 2.0) / np.tan(np.deg2rad(s["fov"] / 2.0))
        return {"fx": f, "fy": f, "cx": s["res"][0]/2, "cy": s["res"][1]/2, "width": s["res"][0], "height": s["res"][1]}

    def _colorize_semantic(self, mask):
        if mask is None: return np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        color = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cid, bgr in self.PALETTE.items():
            color[mask == cid] = bgr
        return color

    def capture_all(self):
        self.update_dynamic_cameras()
        if not self._mappings_built:
            self._build_mappings()
            self._mappings_built = True

        payload = {}
        for name, cam in self.cameras.items():
            try:
                res = cam.render(rgb=True, depth=True, segmentation=True)
                raw_seg = res[2]
                semantic = self._apply_lut(raw_seg, self.sem_lut)
                payload[name] = {
                    "rgb": res[0], 
                    "depth": res[1], 
                    "semantic": semantic,
                    "instance": self._apply_lut(raw_seg, self.inst_lut),
                    "seg_color": self._colorize_semantic(semantic),
                    "intrinsics": self.get_intrinsics(name)
                }
            except Exception as e:
                print(f"Render Error [{name}]: {e}")
                payload[name] = None
        return payload
        
