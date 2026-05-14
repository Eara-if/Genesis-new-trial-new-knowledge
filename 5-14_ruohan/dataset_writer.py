import numpy as np
import torch
import shutil
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

class DatasetWriter:
    def __init__(self, repo_id="genesis_multi_task", root_dir="outputs/lerobot_dataset", fps=50):
        self.repo_id = repo_id
        self.root_dir = Path(root_dir)
        self.fps = fps
        self.episode_idx = 0
        self.dataset = None
        self.current_task = "pick up the box" 
        
        self._init_lerobot_dataset()

    def _init_lerobot_dataset(self):
        features = {
            "observation.images.top_cam": {"dtype": "video", "shape": (3, 480, 640), "names": ["c", "h", "w"]},
            "observation.images.wrist_cam": {"dtype": "video", "shape": (3, 480, 640), "names": ["c", "h", "w"]},
            "observation.tactile.pressure": {"dtype": "float32", "shape": (16, 16), "names": ["h", "w"]},
            "observation.tactile.shear": {"dtype": "float32", "shape": (16, 16), "names": ["h", "w"]},
            "observation.state": {"dtype": "float32", "shape": (6,), "names": ["joint_pos"]},
            "action": {"dtype": "float32", "shape": (4,), "names": ["action_dim"]},
            "reward.dense": {"dtype": "float32", "shape": (1,)},
            "reward.sparse": {"dtype": "float32", "shape": (1,)}
        }

        # [核心修复] 检查是否已经存在有效的数据集，启用追加模式
        # 通过检查底层文件结构(如 meta, data, episodes.jsonl)判断是不是完整数据集
        is_existing = self.root_dir.exists() and (
            (self.root_dir / "meta").exists() or 
            (self.root_dir / "data").exists() or 
            (self.root_dir / "episodes.jsonl").exists()
        )

        if is_existing:
            print(f"🔄 检测到历史数据集 '{self.root_dir}'。启动【追加模式 (Resume)】...")
            try:
                # 抛弃 .create()，直接使用基类加载已存在的数据集
                self.dataset = LeRobotDataset(self.repo_id, root=self.root_dir)
                
                # 获取当前已有的 Episode 数量，无缝衔接序号
                self.episode_idx = self.dataset.num_episodes
                print(f"📊 当前数据集已包含 {self.episode_idx} 个 Episode。本次将从 Episode {self.episode_idx} 开始追加。")
            except Exception as e:
                print(f"❌ 加载历史数据集失败: {e}。由于数据可能已损坏，将清空并重建！")
                shutil.rmtree(self.root_dir)
                is_existing = False

        if not is_existing:
            print(f"✨ 正在 '{self.root_dir}' 创建全新的多模态数据集...")
            if self.root_dir.exists():
                shutil.rmtree(self.root_dir)

            # 只有在确信需要从零开始时，才调用 .create()
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                features=features,
                root=self.root_dir,
                use_videos=True 
            )
            self.episode_idx = 0

    def start_episode(self, metadata=None):
        print(f"📁 开始记录专家轨迹 Episode {self.episode_idx}...")
        self.episode_idx += 1
        if metadata and "task" in metadata:
            self.current_task = metadata["task"]

    def write_step(self, step_idx, payload):
        if self.dataset is None:
            raise RuntimeError("数据集未正确初始化。")

        sensors = payload.get("sensors", {})
        expert_data = payload.get("expert_data", {})

        if not sensors or not expert_data:
            return

        top_rgb = sensors.get("top_cam", {}).get("rgb")
        wrist_rgb = sensors.get("wrist_cam", {}).get("rgb")
        if top_rgb is None: top_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        if wrist_rgb is None: wrist_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        top_tensor = torch.from_numpy(top_rgb.copy()).permute(2, 0, 1).contiguous()
        wrist_tensor = torch.from_numpy(wrist_rgb.copy()).permute(2, 0, 1).contiguous()

        j_pos = np.array(expert_data.get("joint_pos", np.zeros(6)), dtype=np.float32)
        delta_p = np.array(expert_data.get("delta_pos", np.zeros(3)), dtype=np.float32)
        grip_st = np.array([1.0 if expert_data.get("gripper_attached", False) else 0.0], dtype=np.float32)
        action_vector = np.concatenate([delta_p, grip_st], axis=0)

        tactile_info = expert_data.get("tactile", {})
        pressure_map = np.array(tactile_info.get("pressure_map", np.zeros((16, 16))), dtype=np.float32)
        shear_map = np.array(tactile_info.get("shear_map", np.zeros((16, 16))), dtype=np.float32)

        dense_r = expert_data.get("reward_dense", 0.0)
        sparse_r = expert_data.get("reward_sparse", 0.0)

        frame = {
            "observation.images.top_cam": top_tensor,
            "observation.images.wrist_cam": wrist_tensor,
            "observation.tactile.pressure": torch.from_numpy(pressure_map.copy()),
            "observation.tactile.shear": torch.from_numpy(shear_map.copy()),
            "observation.state": torch.from_numpy(j_pos.copy()),
            "action": torch.from_numpy(action_vector.copy()),
            "reward.dense": torch.tensor([dense_r], dtype=torch.float32),
            "reward.sparse": torch.tensor([sparse_r], dtype=torch.float32),
            "task": self.current_task 
        }

        self.dataset.add_frame(frame)

    def end_episode(self):
        self.dataset.save_episode()
        print(f"✅ Episode {self.episode_idx - 1} 保存完毕。")

    def finish_dataset(self):
        # 只要代码不崩溃，LeRobot 会在后台自动完成 parquet 文件落盘
        print(f"🚀 数据集文件流已安全关闭。输出路径: {self.root_dir}")