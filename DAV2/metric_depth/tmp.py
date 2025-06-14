import os
import cv2
from moviepy.editor import ImageSequenceClip
import numpy as np


# 讀取 16 位深度圖
# img_path = '../datasets/tum_rgbd/rgbd_dataset_freiburg3_long_office_household/depth/1341847980.723020.png'
img_path = '../outputs/0111_1/disp/'
img_names = sorted([f for f in os.listdir(img_path) if f.endswith('.png')], key=lambda x: (int(x.split('_')[0][1:])))

all_image_paths = [os.path.join(img_path, img) for img in img_names]

# 獲取圖像的寬度和高度
height, width = 480,640

# 計算每個區域的寬度和高度
region_width = width // 4
region_height = height // 8

# 初始化存儲每個區域中心點深度值的列表

for each_image_path in all_image_paths:
    center_depth_values = []
    depth_image = cv2.imread(each_image_path, cv2.IMREAD_UNCHANGED)
    # 遍歷每個區域
    for i in range(8):
        for j in range(4):
            # 計算當前區域的起始和結束位置
            start_x = j * region_width
            start_y = i * region_height
            end_x = start_x + region_width
            end_y = start_y + region_height
            # 提取當前區域
            region = depth_image[start_y:end_y, start_x:end_x]
            # 計算當前區域中心點的位置
            center_x = start_x + region_width // 2
            center_y = start_y + region_height // 2
            # 獲取中心點的深度值
            center_depth_value = depth_image[center_y, center_x]
            # 將中心點深度值添加到列表中
            center_depth_values.append(center_depth_value)
    # 設置 RGBD.DepthMapFactor
    depth_map_factor = 2000.0  # 根據實際情況設置
    # 將深度值轉換為實際距離（以米為單位）
    center_depth_values_in_meters = [value / depth_map_factor for value in center_depth_values]
    # 打印每個區域中心點的深度值（像素值和米）
    for idx, (pixel_value, meter_value) in enumerate(zip(center_depth_values, center_depth_values_in_meters)):
        print(f"區域 {idx + 1} 的中心點深度值（像素值）: {pixel_value}")
        print(f"區域 {idx + 1} 的中心點深度（米）: {meter_value}")
        # 在圖像上標記每個區域中心點的深度值
        for idx, (pixel_value, meter_value) in enumerate(zip(center_depth_values, center_depth_values_in_meters)):
            center_x = (idx % 4) * region_width + region_width // 2
            center_y = (idx // 4) * region_height + region_height // 2
            # 在中心點畫圓點
            cv2.circle(depth_image, (center_x, center_y), 5, (0, 0, 0), -1)
            # 在右邊標註深度值
            text = f"{meter_value:.2f}m"
            cv2.putText(depth_image, text, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # 保存標記後的圖像
        output_path = "../depth_eval/"+each_image_path.split('/')[-3]
        os.makedirs(output_path, exist_ok=True)
        output_img = output_path +"/" +each_image_path.split('/')[-1].replace('.png','_sign.png')
        cv2.imwrite(output_img, depth_image)
    print(f"標記後的圖像已保存到: {output_img}")