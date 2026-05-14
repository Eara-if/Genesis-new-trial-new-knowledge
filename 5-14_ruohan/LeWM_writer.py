import h5py
import numpy as np
import os
from pathlib import Path

class DatasetWriter:
    """
    专门为 LeWorldModel (LeWM) 打造的数据集生成器
    已适配带有 VacuumGripper (吸盘)、多视角相机和 Tactile 触觉传感器的系统
    """
    def __init__(self, repo_id="genesis_multi_task", root_dir="outputs/lewm_dataset", fps=50):
        self.dataset_name = repo_id
        self.root_dir = Path(os.path.expanduser(root_dir))
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        self.file_path = self.root_dir / f"{self.dataset_name}.h5"
        self.episode_count = 0
        self.total_steps = 0
        self.fps = fps
        
        # 缓存当前 Episode 的所有多模态数据
        self.current_episode_buffer = {
            "pixels": [],           # 默认使用 top_cam
            "action": [],           # 严格对齐 25 维
            "proprio": [],          # joint_pos (6)
            "joint_vel": [],        # joint_vel (6)
            "tactile_pressure": [], # 触觉压力图 (16x16)
            "tactile_shear": [],    # 触觉剪切图 (16x16)
            "rewards": []           # [dense_reward, sparse_reward]
        }
        
        print(f"📁 LeWM 多模态数据集将保存至: {self.file_path} (FPS: {self.fps})")

    def start_episode(self, metadata=None):
        """开始一个新的专家轨迹"""
        for key in self.current_episode_buffer:
            self.current_episode_buffer[key] = []
        task_name = metadata.get("task", "Unknown Task") if metadata else "Unknown Task"
        print(f"🎬 开始记录 Episode {self.episode_count} | 任务: {task_name}")

    def write_step(self, step_idx, payload):
        """写入一帧多模态数据"""
        sensors = payload.get("sensors", {})
        expert_data = payload.get("expert_data", {})

        # 1. 图像处理 (默认取 top_cam，如需多视角可扩展)
        rgb = sensors.get("top_cam", {}).get("rgb")
        if rgb is None: 
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        self.current_episode_buffer["pixels"].append(rgb)

        # 2. 动作 (Action) - 吸盘逻辑
        # 3维位移 + 1维吸盘吸附状态
        delta_p = np.array(expert_data.get("delta_pos", np.zeros(3)), dtype=np.float32)
        # 真值判断：如果 attached 为 True，状态为 1.0；否则为 0.0
        suction_st = np.array([1.0 if expert_data.get("gripper_attached", False) else 0.0], dtype=np.float32)
        base_action = np.concatenate([delta_p, suction_st], axis=0)

        # 强制填充至 25 维，以匹配 LeWM Embedder 的 input_dim=25
        action_vector = np.zeros(25, dtype=np.float32)
        action_vector[:4] = base_action
        self.current_episode_buffer["action"].append(action_vector)

        # 3. 本体感知与运动学状态
        j_pos = np.array(expert_data.get("joint_pos", np.zeros(6)), dtype=np.float32)
        j_vel = np.array(expert_data.get("joint_vel", np.zeros(6)), dtype=np.float32)
        self.current_episode_buffer["proprio"].append(j_pos)
        self.current_episode_buffer["joint_vel"].append(j_vel)

        # 4. 触觉反馈 (Tactile)
        tactile_data = expert_data.get("tactile", {})
        pressure_map = tactile_data.get("pressure_map", np.zeros((16, 16), dtype=np.float32))
        shear_map = tactile_data.get("shear_map", np.zeros((16, 16), dtype=np.float32))
        self.current_episode_buffer["tactile_pressure"].append(pressure_map)
        self.current_episode_buffer["tactile_shear"].append(shear_map)

        # 5. 双重奖励机制 (Rewards)
        dense_r = expert_data.get("reward_dense", 0.0)
        sparse_r = expert_data.get("reward_sparse", 0.0)
        self.current_episode_buffer["rewards"].append(np.array([dense_r, sparse_r], dtype=np.float32))

        self.total_steps += 1

    def end_episode(self):
        """将当前缓存的 Episode 写入 HDF5 文件"""
        if not self.current_episode_buffer["pixels"]:
            return

        # 转换为 numpy 数组
        data_to_save = {
            "pixels": np.array(self.current_episode_buffer["pixels"], dtype=np.uint8),
            "action": np.array(self.current_episode_buffer["action"], dtype=np.float32),
            "proprio": np.array(self.current_episode_buffer["proprio"], dtype=np.float32),
            "joint_vel": np.array(self.current_episode_buffer["joint_vel"], dtype=np.float32),
            "tactile_pressure": np.array(self.current_episode_buffer["tactile_pressure"], dtype=np.float32),
            "tactile_shear": np.array(self.current_episode_buffer["tactile_shear"], dtype=np.float32),
            "rewards": np.array(self.current_episode_buffer["rewards"], dtype=np.float32),
        }
        
        num_steps = len(data_to_save["pixels"])
        data_to_save["episode_idx"] = np.full(num_steps, self.episode_count, dtype=np.int64)
        data_to_save["step_idx"] = np.arange(num_steps, dtype=np.int64)

        # 以追加模式打开 HDF5
        mode = 'a' if self.file_path.exists() else 'w'
        with h5py.File(self.file_path, mode) as f:
            for key, data in data_to_save.items():
                if key not in f:
                    # 创建动态维度的数据集 (第一维为 None，支持无限追加)
                    max_shape = (None,) + data.shape[1:]
                    f.create_dataset(key, data=data, maxshape=max_shape, chunks=True)
                else:
                    # 扩展维度并追加数据
                    f[key].resize((f[key].shape[0] + data.shape[0]), axis=0)
                    f[key][-data.shape[0]:] = data

        print(f"✅ Episode {self.episode_count} 保存完毕。当前局步数: {num_steps} | 总步数: {self.total_steps}")
        self.episode_count += 1

    def finish_dataset(self):
        print(f"🚀 数据集生成完毕！已封装多模态特征 (视觉/触觉/奖励/动作)。文件名: {self.file_path.name}")