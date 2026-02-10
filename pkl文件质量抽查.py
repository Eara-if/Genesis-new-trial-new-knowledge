import pickle
import cv2
import numpy as np
import os

# 1. 动态获取路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 自动寻找文件夹下最新的 pkl 文件
files = [f for f in os.listdir(current_dir) if f.endswith('.pkl')]
files.sort(key=lambda x: os.path.getmtime(os.path.join(current_dir, x)))
filename = files[-1]  # 获取最新生成的文件

print(f">>> 正在读取文件: {filename}")

with open(os.path.join(current_dir, filename), "rb") as f:
    data = pickle.load(f)

# 2. 设置视频写入器
height, width, _ = data[0]["observation"]["rgb"].shape
video_path = os.path.join(current_dir, "trajectory_preview.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 设定 30 FPS，这样 4920 帧大概是 2.5 分钟的视频
video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

print(">>> 正在合成视频...")
for i, frame in enumerate(data):
    rgb_img = frame["observation"]["rgb"]
    # Genesis 输出的是 RGB，OpenCV 需要 BGR
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    video_writer.write(bgr_img)
    
    if i % 500 == 0:
        print(f"进度: {i}/{len(data)}")

video_writer.release()
print(f">>> 成功！视频已保存至: {video_path}")