import cv2
import numpy as np
import torch
import torchvision.transforms as T
import timm

class VisionProcessor:
    def __init__(self):
        # --- 基础参数 ---
        self.res_x, self.res_y = 640, 480
        self.cx, self.cy = self.res_x / 2.0, self.res_y / 2.0
        self.fy = (self.res_y / 2.0) / np.tan(np.deg2rad(50.0 / 2.0))
        self.fx = self.fy 
        self.cam_x, self.cam_y, self.cam_z = 0.5, 0.0, 1.2

        self.ROI_X_MIN, self.ROI_X_MAX = 0.35, 0.75
        self.ROI_Y_MIN, self.ROI_Y_MAX = -0.40, 0.40
        self.DEPTH_GRAD_THRESH = 0.03 # 历史代码：深度突变阈值

        # --- DINOv2 潜空间特征提取器 ---
        self.device = torch.device("cpu")
        print(f"🧠 [Vision] 正在通过 timm 加载 DINOv2 (设备: {self.device})...")

        try:
            self.dinov2 = timm.create_model(
                'vit_small_patch14_dinov2', 
                pretrained=True, 
                num_classes=0
            ).to(self.device)
            self.dinov2.eval()
            print("✅ DINOv2 潜空间编码器准备就绪。")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((518, 518)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_latent_features(self, rgb_image):
        if rgb_image is None or not hasattr(self, 'dinov2'): return None
        img_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.dinov2(img_tensor)
        return features.squeeze(0).cpu().numpy()

    def evaluate_feature_divergence(self, feat_global, feat_wrist):
        if feat_global is None or feat_wrist is None: return 0.0
        cos_sim = np.dot(feat_global, feat_wrist) / (np.linalg.norm(feat_global) * np.linalg.norm(feat_wrist))
        return float(cos_sim)

    def get_bev_and_data(self, rgb_image, depth_image):
        if rgb_image is None or depth_image is None: return None, []

        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # === 1. 利用深度断崖分离多个箱子 ===
        base_mask = np.where(depth_image < 1.19, 255, 0).astype(np.uint8)
        
        grad_x = cv2.Sobel(depth_image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_image, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        
        depth_edges = np.where(grad_mag > self.DEPTH_GRAD_THRESH, 255, 0).astype(np.uint8)
        depth_edges = cv2.dilate(depth_edges, np.ones((3, 3), np.uint8), iterations=1)
        
        separated_mask = cv2.bitwise_and(base_mask, cv2.bitwise_not(depth_edges))
        final_mask = cv2.morphologyEx(separated_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_objects = []

        # === 2. 终极工业级混合裁决算法 (Hybrid Decision) ===
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 150 < area < 15000:
                # [核心策略 A]：获取纯物理几何中心 (完美适用于完整倾斜箱子)
                rect = cv2.minAreaRect(cnt)
                geom_center_x, geom_center_y = int(rect[0][0]), int(rect[0][1])

                mask = np.zeros_like(final_mask)
                cv2.drawContours(mask, [cnt], -1, 255, -1)

                # [核心策略 B]：获取距离变换，评估安全度
                dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)

                if max_val < 5.0: 
                    continue # 太细长的缝隙，放弃

                # [核心裁决]：交叉验证
                # 安全获取几何中心点在 mask 内部的距离值（防止越界）
                if 0 <= geom_center_x < self.res_x and 0 <= geom_center_y < self.res_y:
                    dist_at_geom = dist_transform[geom_center_y, geom_center_x]
                else:
                    dist_at_geom = 0.0

                # 裁决逻辑：如果几何中心的安全半径达到了最大内切圆半径的 60% 以上
                # 说明它是一个完整的或者只被轻微遮挡的箱子，坚决优先使用几何中心！
                if dist_at_geom > max_val * 0.6:
                    target_u, target_v = geom_center_x, geom_center_y
                    color = (255, 0, 0) # 蓝色准心：代表启动了几何中心模式
                else:
                    # 发生了严重的 L 型/C 型遮挡，几何中心悬空了！
                    # 降级模式：提取所有接近最大值的安全区域，求它们的重心，彻底解决长条偏移问题
                    safe_zone_mask = (dist_transform > max_val * 0.8).astype(np.uint8)
                    M = cv2.moments(safe_zone_mask)
                    if M["m00"] != 0:
                        target_u = int(M["m10"] / M["m00"])
                        target_v = int(M["m01"] / M["m00"])
                    else:
                        target_u, target_v = max_loc
                    color = (0, 165, 255) # 橙色准心：代表启动了降级防遮挡模式

                # 提取抓取点 3x3 局部的平滑深度，避免单点噪点
                local_depths = depth_image[max(0, target_v-1):target_v+2, max(0, target_u-1):target_u+2]
                target_depth = np.mean(local_depths)

                # === Debug 可视化 ===
                # 画出外接矩形（绿色）
                box_pts = np.int32(cv2.boxPoints(rect))
                cv2.drawContours(bgr, [box_pts], 0, (0, 255, 0), 1)
                # 画出最终决定的准心
                cv2.circle(bgr, (target_u, target_v), 4, color, -1)

                # === 坐标系转换 ===
                dx_img = (target_u - self.cx) * target_depth / self.fx
                dy_img = (target_v - self.cy) * target_depth / self.fy
                world_x, world_y = self.cam_x + dx_img, self.cam_y - dy_img 
                
                if self.ROI_X_MIN < world_x < self.ROI_X_MAX and self.ROI_Y_MIN < world_y < self.ROI_Y_MAX:
                    precise_z = self.cam_z - target_depth - 0.002
                    detected_objects.append({'pos': np.array([world_x, world_y, precise_z])})
                    
        # 全局排序：无论找到多少个可抓取面，始终让机械臂去抓绝对海拔最高（Z最大）的那一个
        detected_objects.sort(key=lambda item: item['pos'][2], reverse=True)
        return bgr, detected_objects