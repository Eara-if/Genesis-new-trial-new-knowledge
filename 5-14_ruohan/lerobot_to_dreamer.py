import os
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

os.environ["HF_HUB_OFFLINE"] = "1"

LEROBOT_DATA_DIR = "outputs/lerobot_dataset"
DREAMER_OUT_DIR = "data_processed_dreamerv3"
os.makedirs(DREAMER_OUT_DIR, exist_ok=True)

def convert_lerobot_to_dreamer():
    print(f"🔄 正在从 {LEROBOT_DATA_DIR} 加载 LeRobot 数据集...")
    
    try:
        dataset = LeRobotDataset("genesis_multi_task", root=LEROBOT_DATA_DIR)
    except Exception as e:
        print(f"❌ 加载失败，请检查数据集路径。错误: {e}")
        return

    hf_dataset = dataset.hf_dataset
    episodes_array = np.array(hf_dataset['episode_index'])
    unique_episodes = np.unique(episodes_array)
    
    print(f"📊 发现 {len(unique_episodes)} 个 Episodes，开始转换为多流 (Multi-stream) 格式...")

    for ep_idx in unique_episodes:
        frame_indices = np.where(episodes_array == ep_idx)[0]
        ep_len = len(frame_indices)
        
        # 拆分所有模态的 Buffer
        images_top = []
        images_wrist = []
        tactile_pressures = []
        tactile_shears = []
        actions = []
        states = []
        rewards_combined = []
        
        for idx in frame_indices:
            frame = dataset[int(idx)]
            
            # 视觉流 1：上帝视角
            top_tensor = frame["observation.images.top_cam"]
            top_np = (top_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            images_top.append(top_np)
            
            # 视觉流 2：腕部视角
            wrist_tensor = frame["observation.images.wrist_cam"]
            wrist_np = (wrist_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            images_wrist.append(wrist_np)

            # 触觉流
            tactile_pressures.append(frame["observation.tactile.pressure"].numpy().astype(np.float32))
            tactile_shears.append(frame["observation.tactile.shear"].numpy().astype(np.float32))
            
            # 动作与本体
            act = frame["action"].numpy()
            delta_pos = act[:3] / 0.007 
            gripper_st = act[3:]
            norm_action = np.concatenate([delta_pos, gripper_st])
            norm_action = np.clip(norm_action, -1.0, 1.0)
            actions.append(norm_action.astype(np.float32))
            
            states.append(frame["observation.state"].numpy().astype(np.float32))
            # [安全读取] 使用 .item() 安全提取标量，兼容 0D 或 1D 张量
            if "reward.dense" in frame and "reward.sparse" in frame:
                r_dense = frame["reward.dense"].item()
                r_sparse = frame["reward.sparse"].item()
                r_total = (0.1 * r_dense) + (1.0 * r_sparse)
            else:
                # 兼容旧版数据集
                r_total = 1.0 if int(idx) == frame_indices[-1] else 0.0
                
            rewards_combined.append(r_total)
            rewards_combined.append(r_total)
        # [新增] 转换为 numpy 数组
        rewards = np.array(rewards_combined, dtype=np.float32)

        is_first = np.zeros(ep_len, dtype=bool)
        is_first[0] = True
        is_terminal = np.zeros(ep_len, dtype=bool)
        is_terminal[-1] = True  
        discount = 1.0 - is_terminal.astype(np.float32)

        # 核心修改：不再做粗暴的图像拼接，保持数据独立性
        dreamer_data = {
            'image_top': np.stack(images_top),             # [T, 480, 640, 3]
            'image_wrist': np.stack(images_wrist),         # [T, 480, 640, 3]
            'tactile_pressure': np.stack(tactile_pressures), # [T, 16, 16]
            'tactile_shear': np.stack(tactile_shears),       # [T, 16, 16]
            'action': np.stack(actions),
            'state': np.stack(states),
            'reward': rewards,
            'is_first': is_first,
            'is_terminal': is_terminal,
            'discount': discount
        }

        save_path = os.path.join(DREAMER_OUT_DIR, f"episode_{ep_idx:04d}.npz")
        np.savez_compressed(save_path, **dreamer_data)
        print(f"✅ 已生成: {save_path} (包含独立双视觉与双触觉通道, 长度: {ep_len} 步)")

if __name__ == "__main__":
    convert_lerobot_to_dreamer()