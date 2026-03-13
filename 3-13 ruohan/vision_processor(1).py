import cv2
import numpy as np

class VisionProcessor:
    def __init__(self):
        # 物理相机内参
        self.res_x = 640
        self.res_y = 480
        self.cx = self.res_x / 2.0
        self.cy = self.res_y / 2.0
        
        # 焦距计算 (FOV=50)
        self.fy = (self.res_y / 2.0) / np.tan(np.deg2rad(50.0 / 2.0))
        self.fx = self.fy 
        
        self.cam_x = 0.5
        self.cam_y = 0.0
        self.cam_z = 1.2

        # 严格收紧的工业级电子围栏 (Pick Zone)
        self.ROI_X_MIN = 0.35
        self.ROI_X_MAX = 0.75
        self.ROI_Y_MIN = -0.40
        self.ROI_Y_MAX = 0.40
        
        # 【新增】深度梯度阈值 (物理单位：米)
        # 0.03表示：如果相邻像素的深度(高度)突变超过 3cm，则认定为不同物体的交界边缘
        self.DEPTH_GRAD_THRESH = 0.03

    def get_bev_and_data(self, rgb_image, depth_image):
        if rgb_image is None or depth_image is None:
            return None, []

        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # 1. 基础工作区截断 (Z方向二值化，获取所有潜在物体的轮廓)
        base_mask = np.where(depth_image < 1.19, 255, 0).astype(np.uint8)
        
        # 2. 深度体积梯度计算
        # 使用 Sobel 算子提取深度图在X和Y方向的一阶导数 (要求 depth_image 为物理米数)
        grad_x = cv2.Sobel(depth_image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_image, cv2.CV_32F, 0, 1, ksize=3)
        
        # 计算梯度幅值
        grad_mag = cv2.magnitude(grad_x, grad_y)
        
        # 提取高度突变的边缘
        depth_edges = np.where(grad_mag > self.DEPTH_GRAD_THRESH, 255, 0).astype(np.uint8)
        
        # 膨胀这些边缘，确保它们有足够的厚度去彻底切断粘连的 mask
        edge_kernel = np.ones((3, 3), np.uint8)
        depth_edges = cv2.dilate(depth_edges, edge_kernel, iterations=1)
        
        # 3. 从基础 Mask 中抠除高度突变边缘，物理分离紧密排列的不同高度物体
        separated_mask = cv2.bitwise_and(base_mask, cv2.bitwise_not(depth_edges))
        
        # 4. 形态学开运算去噪
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(separated_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_objects = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            rect = cv2.minAreaRect(cnt)
            (u, v), (w, h), angle = rect
            u, v = int(u), int(v)

            if area < 200:
                continue 
            if area > 15000:
                # 过大的色块描成黄色作为Debug警告
                cv2.drawContours(bgr, [np.int32(cv2.boxPoints(rect))], 0, (0, 255, 255), 2)
                continue 
            
            u_c = np.clip(u, 2, self.res_x - 3)
            v_c = np.clip(v, 2, self.res_y - 3)
            local_depths = depth_image[v_c-2:v_c+3, u_c-2:u_c+3]
            obj_depth = np.median(local_depths)
            
            dx_img = (u - self.cx) * obj_depth / self.fx
            dy_img = (v - self.cy) * obj_depth / self.fy
            
            # Mode 3 混合符号系映射
            world_x = self.cam_x + dx_img 
            world_y = self.cam_y - dy_img 
            
            # ROI 越界拦截 
            if not (self.ROI_X_MIN < world_x < self.ROI_X_MAX and self.ROI_Y_MIN < world_y < self.ROI_Y_MAX):
                cv2.drawContours(bgr, [np.int32(cv2.boxPoints(rect))], 0, (0, 0, 255), 2)
                continue
            
            precise_z = self.cam_z - obj_depth - 0.002
            
            detected_objects.append({
                'pos': np.array([world_x, world_y, precise_z]),
                'pixel': (u, v),
                'depth': obj_depth
            })
            
            cv2.drawContours(bgr, [np.int32(cv2.boxPoints(rect))], 0, (0, 255, 0), 2)
            cv2.circle(bgr, (u, v), 3, (0, 255, 0), -1)
            cv2.putText(bgr, f"Z:{precise_z:.2f} X:{world_x:.2f}", (u, v-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        detected_objects.sort(key=lambda item: item['pos'][2], reverse=True)
                        
        return bgr, detected_objects