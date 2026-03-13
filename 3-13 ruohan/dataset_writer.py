import json
from pathlib import Path

import cv2
import numpy as np


class DatasetWriter:
    """统一保存 RGB / depth / 语义分割 / 触觉图 / 元信息。"""

    def __init__(self, root_dir="outputs/dataset", prefix="episode", save_every_n_steps=5):
        self.root_dir = Path(root_dir)
        self.prefix = prefix
        self.save_every_n_steps = max(1, int(save_every_n_steps))
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.episode_dir = None
        self.episode_idx = 0

    def start_episode(self, metadata=None):
        self.episode_dir = self.root_dir / f"{self.prefix}_{self.episode_idx:04d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.episode_idx += 1
        if metadata is not None:
            self._write_json(self.episode_dir / "episode_meta.json", metadata)

    def should_save(self, step_idx):
        return (step_idx % self.save_every_n_steps) == 0

    def write_step(self, step_idx, payload):
        if self.episode_dir is None:
            self.start_episode()

        frame_dir = self.episode_dir / f"frame_{step_idx:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        sensors = payload.get("sensors", {})
        for cam_name, cam_data in sensors.items():
            cam_dir = frame_dir / cam_name
            cam_dir.mkdir(parents=True, exist_ok=True)

            rgb = cam_data.get("rgb")
            if rgb is not None:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(cam_dir / "rgb.png"), bgr)

            depth = cam_data.get("depth")
            if depth is not None:
                np.save(cam_dir / "depth.npy", depth.astype(np.float32))
                depth_vis = self._depth_to_vis(depth)
                cv2.imwrite(str(cam_dir / "depth_vis.png"), depth_vis)

            semantic = cam_data.get("semantic")
            if semantic is not None:
                np.save(cam_dir / "semantic.npy", semantic.astype(np.uint8))

            instance = cam_data.get("instance")
            if instance is not None:
                np.save(cam_dir / "instance.npy", instance.astype(np.int32))

            seg_color = cam_data.get("seg_color")
            if seg_color is not None:
                cv2.imwrite(str(cam_dir / "seg_color.png"), seg_color)

            intr = cam_data.get("intrinsics")
            extr = cam_data.get("extrinsics")
            if intr is not None or extr is not None:
                self._write_json(
                    cam_dir / "camera_meta.json",
                    {"intrinsics": intr, "extrinsics": extr}
                )

        tactile = payload.get("tactile")
        if tactile is not None:
            tactile_dir = frame_dir / "tactile"
            tactile_dir.mkdir(parents=True, exist_ok=True)

            pressure_map = tactile.get("pressure_map")
            shear_map = tactile.get("shear_map")
            contact_mask = tactile.get("contact_mask")

            if pressure_map is not None:
                np.save(tactile_dir / "pressure_map.npy", pressure_map.astype(np.float32))
                cv2.imwrite(str(tactile_dir / "pressure_map.png"), self._heatmap_to_vis(pressure_map))

            if shear_map is not None:
                np.save(tactile_dir / "shear_map.npy", shear_map.astype(np.float32))
                cv2.imwrite(str(tactile_dir / "shear_map.png"), self._heatmap_to_vis(shear_map))

            if contact_mask is not None:
                np.save(tactile_dir / "contact_mask.npy", contact_mask.astype(np.float32))
                cv2.imwrite(str(tactile_dir / "contact_mask.png"), (contact_mask * 255).astype(np.uint8))

            if tactile.get("meta") is not None:
                self._write_json(tactile_dir / "tactile_meta.json", tactile["meta"])

        if payload.get("meta") is not None:
            self._write_json(frame_dir / "meta.json", payload["meta"])

    def _write_json(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._to_serializable(data), f, ensure_ascii=False, indent=2)

    def _to_serializable(self, value):
        if isinstance(value, dict):
            return {k: self._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    def _depth_to_vis(self, depth):
        depth = np.asarray(depth, dtype=np.float32)
        finite_mask = np.isfinite(depth)
        if not finite_mask.any():
            return np.zeros((*depth.shape, 3), dtype=np.uint8)

        valid = depth[finite_mask]
        d_min, d_max = float(valid.min()), float(valid.max())
        if abs(d_max - d_min) < 1e-6:
            norm = np.zeros_like(depth, dtype=np.uint8)
        else:
            norm = ((depth - d_min) / (d_max - d_min) * 255.0).clip(0, 255).astype(np.uint8)

        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    def _heatmap_to_vis(self, heatmap):
        heatmap = np.asarray(heatmap, dtype=np.float32)
        norm = np.clip(heatmap, 0.0, 1.0)
        norm = (norm * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)