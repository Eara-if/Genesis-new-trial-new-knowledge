import numpy as np
import os
import cv2

class DreamerDataCollector:
    def __init__(self, save_dir="./data/training_episodes", target_size=(64, 64)):
        self.save_dir = save_dir
        self.target_size = target_size
        os.makedirs(save_dir, exist_ok=True)
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {
            'image': [], 'action': [], 'state': [], 'reward': [], 
            'is_first': [], 'is_last': [], 'is_terminal': []
        }
    def add_step(self, raw_img, action, state, reward=0, done=False, is_first=False):
        # 强制将输入转换为 numpy 数组，如果是元组则取第一个元素
        if isinstance(raw_img, tuple):
            raw_img = raw_img[0]
            
        # 确保它是 numpy 数组
        raw_img = np.asarray(raw_img)
        
        # 剩下的逻辑不变...
        if raw_img.shape[:2] != self.target_size:
            # 注意：cv2.resize 期望的是 (W, H)，而 numpy shape 是 (H, W)
            # 为了保险，显式指定目标尺寸
            resized_img = cv2.resize(raw_img, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)
        else:
            resized_img = raw_img
        
        # --- 核心修复：确保 action 和 state 都转为 CPU 上的 Numpy ---
        def to_numpy(data):
            # 如果是 torch 张量，先转 CPU 再转 Numpy
            if hasattr(data, 'cpu'):
                return data.cpu().numpy()
            return np.array(data)

        # 存入 Buffer 前统一转换
        self.buffer['action'].append(to_numpy(action).astype(np.float32))
        self.buffer['state'].append(to_numpy(state).astype(np.float32))
        # 2. 存入 Buffer
        self.buffer['image'].append(resized_img.astype(np.uint8))
        self.buffer['reward'].append(np.float32(reward))
        self.buffer['is_first'].append(bool(is_first))
        self.buffer['is_last'].append(bool(done))
        self.buffer['is_terminal'].append(bool(done))

    # 统一命名为 save_episode
    def save_episode(self, episode_idx):
        if not self.buffer['image']: 
            print("警告: Buffer 为空，跳过保存。")
            return
            
        path = os.path.join(self.save_dir, f'episode_{episode_idx:04d}.npz')
        # 修复了原代码中 save_data 变量名的错误
        save_dict = {k: np.array(v) for k, v in self.buffer.items()}
        np.savez_compressed(path, **save_dict)
        print(f"成功保存 Episode {episode_idx} 至 {path}")
        self.reset_buffer()