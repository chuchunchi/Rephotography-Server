import os
from skimage import io
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math

# our model
# from flow_estimator import Flow_estimator
# from config import get_twins_args, get_eval_args, get_life_args

torch.set_grad_enabled(False)
import flow_vis
import time
# from colormap import get_colormap
# from eval_utils import resize_lighting, resize_viewpoint
# from metrics import metrics

def optimize_alierr(points_images1, points_images2, al_err, shape_):
    height, width = shape_
    center = (width / 2.0, height / 2.0)
    # print('center: ' + str(center))
    k = int(al_err / 10)  # shift pixels
    direct_npts = {}
    # print(type(points_images1))
    # print('points_images1:', points_images1)
    # print('points_images1-k:', points_images1[1]-k)
    # print('points_images2:', points_images2)
    shift = k * (points_images1 - np.expand_dims(center, axis=1)) / np.linalg.norm(points_images1 - np.expand_dims(center, axis=1), axis=0)
    
    points_image1_forward = points_images1 + shift
    direct_npts['forward'] = points_image1_forward

    points_image1_back = points_images1 - shift
    direct_npts['back'] = points_image1_back

    points_image1_up = points_images1.copy()
    points_image1_up[1] = points_image1_up[1] + k
    direct_npts['up'] = points_image1_up

    points_image1_down = points_images1.copy()
    points_image1_down[1] = points_image1_down[1] - k
    direct_npts['down'] = points_image1_down

    points_image1_right = points_images1.copy()
    points_image1_right[0] = points_image1_right[0] - k
    direct_npts['right'] = points_image1_right

    points_image1_left = points_images1.copy()
    points_image1_left[0] = points_image1_left[0] + k
    direct_npts['left'] = points_image1_left

    # print('points_images1:', points_images1)
    # print('points_images2:', points_images2)
    # print(direct_npts)

    return direct_npts

def visual_flow_matching(coord1_flow_2D_norm_i, coord2_flow_2D_norm_i, ref_cv, tar_cv, save_door):
    coord1_flow_2D_i_width = coord1_flow_2D_norm_i[0][0].cpu().numpy()
    coord1_flow_2D_i_height = coord1_flow_2D_norm_i[0][1].cpu().numpy()

    coord2_flow_2D_i_width = coord2_flow_2D_norm_i[0][0].cpu().numpy()
    coord2_flow_2D_i_height = coord2_flow_2D_norm_i[0][1].cpu().numpy()
    
    # Draw the points on the image
    radius = 2
    red = (0, 0, 255)  # Red
    green = (0, 255, 0)
    blue = (255, 0, 0)
    thickness = -1  # Filled circle
    
    ref_cv = cv2.cvtColor(ref_cv, cv2.COLOR_RGB2BGR)
    tar_cv = cv2.cvtColor(tar_cv, cv2.COLOR_RGB2BGR)
    # ref_cv = cv2.convertScaleAbs(ref_cv)
    # tar_bgr = cv2.convertScaleAbs(tar_cv)
    # print(ref_cv.shape)
    height, width,_ = ref_cv.shape

    concatenated_image = np.concatenate((ref_cv, tar_cv), axis=1)

    for i in range(len(coord1_flow_2D_i_width)):

        
        ref_x = int(coord1_flow_2D_i_width[i])
        ref_y = int(coord1_flow_2D_i_height[i])
        tar_x = int(coord2_flow_2D_i_width[i])
        tar_y = int(coord2_flow_2D_i_height[i])
        # print('tar_x:', tar_x)
        # print('tar_y:', tar_y)
        # if ref_x < 0 or ref_y < 0 or tar_x < 0 or tar_y < 0:
        #     continue
        # if ref_x > width or ref_y > height or tar_x > width or tar_y > height:
        #     continue
        cv2.circle(ref_cv, (ref_x, ref_y), radius, blue, thickness)
        cv2.circle(tar_cv, (tar_x, tar_y), radius, red, thickness)
        cv2.circle(concatenated_image, (ref_x, ref_y), radius, blue, thickness)
        cv2.circle(concatenated_image, (tar_x + width, tar_y), radius, red, thickness)
        cv2.line(concatenated_image, (ref_x, ref_y), (tar_x + width, tar_y), green, 1)

    cv2.imwrite(os.path.join(save_door, "concatenated_image_with_keypoints_flow.png"), concatenated_image)
    cv2.imwrite(os.path.join(save_door, "current_image_with_keypoints_flow.png"), ref_cv)
    cv2.imwrite(os.path.join(save_door, "target_image_with_keypoints_flow.png"), tar_cv)

# def save_visualization(coord1, coord2, al_err, dir, ref_cv, tar_cv, output_path=None):
#     """
#     將座標繪製在參考圖像上進行視覺化並將結果存為圖片。

#     參數：
#     - coord1：座標1的陣列，形狀為 (2, N)
#     - coord2：座標2的陣列，形狀為 (2, N)
#     - ref_image_path：參考圖像的路徑
#     - output_path：輸出圖片的路徑

#     返回值：
#     - 無
#     """

#     # 讀取參考圖像
#     ref_bgr = cv2.cvtColor(ref_cv, cv2.COLOR_RGB2BGR)
#     tar_bgr = cv2.cvtColor(tar_cv, cv2.COLOR_RGB2BGR)

#      # 確保兩張照片的尺寸相同
#     height, width, _ = ref_bgr.shape

#     tar_bgr = cv2.resize(tar_bgr, (width, height))
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.7
#     color = (0, 255, 0)  # Green color in BGR format
#     thickness = 2

#     # 將兩張照片疊加在一起
#     blended_image = cv2.addWeighted(ref_bgr, 0.5, tar_bgr, 0.5, 0)
#     cv2.putText(blended_image, dir +" "+ str(al_err) , (10, 50), font, font_scale, color, thickness)
#     # 繪製座標1
#     for i in range(coord1.shape[1]):
#         cv2.circle(blended_image, (int(coord1[0, i]), int(coord1[1, i])), 3, (255, 0, 0), -1)

#     # 繪製座標2
#     for i in range(coord2.shape[1]):
#         cv2.circle(blended_image, (int(coord2[0, i]), int(coord2[1, i])), 3, (0, 0, 255), -1)

#     # 連線對應點
#     for i in range(coord1.shape[1]):
#         cv2.line(blended_image, (int(coord1[0, i]), int(coord1[1, i])),
#                  (int(coord2[0, i]), int(coord2[1, i])), (0, 255, 0), 1)
        
#     # 儲存圖片
#     # if output_path is not None:
#     cv2.imwrite(output_path, blended_image)

#     return blended_image

def save_visualization(coord1, coord2, al_err, dir, ref_cv, tar_cv, output_path=None):
    """
    將座標繪製在參考圖像上進行視覺化並將結果存為圖片。

    參數：
    - coord1：座標1的陣列，形狀為 (2, N)
    - coord2：座標2的陣列，形狀為 (2, N)
    - ref_image_path：參考圖像的路徑
    - output_path：輸出圖片的路徑

    返回值：
    - 無
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 0)  # Green color in BGR format
    thickness = 2

    # 將兩張照片疊加在一起
    blended_image = cv2.addWeighted(ref_cv, 0.5, tar_cv, 0.5, 0)
    cv2.putText(blended_image, dir +" "+ str(al_err) , (10, 50), font, font_scale, color, thickness)
    # 繪製座標1
    for i in range(coord1.shape[1]):
        cv2.circle(blended_image, (int(coord1[0, i]), int(coord1[1, i])), 3, (255, 0, 0), -1)

    # 繪製座標2
    for i in range(coord2.shape[1]):
        cv2.circle(blended_image, (int(coord2[0, i]), int(coord2[1, i])), 3, (0, 0, 255), -1)

    # 連線對應點
    for i in range(coord1.shape[1]):
        cv2.line(blended_image, (int(coord1[0, i]), int(coord1[1, i])),
                 (int(coord2[0, i]), int(coord2[1, i])), (0, 255, 0), 1)
        
    # 儲存圖片
    if output_path is not None:
        cv2.imwrite(output_path, blended_image)

    return blended_image

def compute_alignment_score(coord1, coord2):
    """
    計算 coord1 和 coord2 之間的對齊效果評分。

    參數：
    - coord1：第一組座標的陣列，形狀為 (2, N)
    - coord2：第二組座標的陣列，形狀為 (2, N)

    返回值：
    - 對齊效果評分
    """

    # 計算兩組座標之間的歐氏距離
    coor_err = coord1 - coord2
    distances = np.sqrt(np.sum((coor_err)**2, axis=0))
    
    
    # 計算平均距離的加總作為對齊效果評分
    alignment_score = np.sum(distances) / len(distances)

    return alignment_score, coor_err, distances

def flow2coord(flow):
    """
    Generate flat homogeneous coordinates 1 and 2 from optical flow. 
    Args:
        flow: bx2xhxw, torch.float32
    Output:
        coord1_hom: bx3x(h*w)
        coord2_hom: bx3x(h*w)
    """
    b, _, h, w = flow.size()
    coord1 = torch.zeros_like(flow)
    coord1[:,0,:,:] += torch.arange(w).float().cuda()
    coord1[:,1,:,:] += torch.arange(h).float().cuda()[:, None]
    coord2 = coord1 + flow
    coord1_flat = coord1.reshape(b, 2, h*w)
    coord2_flat = coord2.reshape(b, 2, h*w)

    ones = torch.ones((b, 1, h*w), dtype=torch.float32).cuda()
    coord1_hom = torch.cat((coord1_flat, ones), dim=1)
    coord2_hom = torch.cat((coord2_flat, ones), dim=1)
    return coord1_hom, coord2_hom

# Draw keypoints on an image
def draw_keypoints(image, keypoints):
    # Convert the image to a suitable depth (e.g., 8-bit)
    image = cv2.convertScaleAbs(image)
    # Convert the image to RGB color space
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Convert the image back to BGR color space
    image_with_keypoints_bgr = cv2.cvtColor(image_with_keypoints, cv2.COLOR_RGB2BGR)
    return image_with_keypoints_bgr

def select_direct(al_err, direct_npts, points_images2, blended_image=None, save_door=None):
    direct = 'original'
    new_elsti_alignment_score = al_err
    for direction, new_points in direct_npts.items():
    # 在這裡進行對每個鍵值對的操作
    # 計算兩組座標之間的歐氏距離
        new_coor_err = points_images2 - new_points
        new_distances = np.sqrt(np.sum((new_coor_err)**2, axis=0))
        best_points = new_points
    # 計算平均距離的加總作為對齊效果評分
        new_alignment_score = np.sum(new_distances) / len(new_distances)
        if new_alignment_score < new_elsti_alignment_score:
            new_elsti_alignment_score = new_alignment_score
            direct = direction
    #------------------------save move result-----------------------------------
    # cv2.putText(blended_image, direct + ': ' + str(new_elsti_alignment_score), (10, 100), font, font_scale, color, thickness)
    # # 繪製座標3
    # for i in range(new_points.shape[1]):
    #     cv2.circle(blended_image, (int(best_points[0, i]), int(best_points[1, i])), 3, (255, 255, 0), -1)
    # for i in range(points_images2.shape[1]):
    #         cv2.line(blended_image, (int(points_images2[0, i]), int(points_images2[1, i])),
    #             (int(new_points[0, i]), int(new_points[1, i])), (0, 255, 255), 1)
    #     # 儲存圖片
    # cv2.imwrite(os.path.join(save_door, 'drone_move_' + direction + '.png'), blended_image)
    return direct

def flow_to_error(curr_image, target_image, flow, video_idx, intrinsic, depth_c, depthfactor, save_door=""):#by zxr
    b, _, h, w = flow.size()
    height, width,_ = curr_image.shape

    coord1_flow_2D, coord2_flow_2D = flow2coord(flow)    # Bx3x(H*W) 
    coord1_flow_2D = coord1_flow_2D.view(b,3,h,w)        
    coord2_flow_2D = coord2_flow_2D.view(b,3,h,w)
    min_matches = 20

    PTS1=[]; PTS2=[];                                       # point list

    sift = cv2.SIFT_create()

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    cur_cv = curr_image
    tar_cv = target_image

            # detect key points
    kp1, des1 = sift.detectAndCompute(cur_cv.astype(np.uint8),None)
    kp2, des2 = sift.detectAndCompute(tar_cv.astype(np.uint8),None)
    try:
        # filter out some key points
        matches = flann.knnMatch(des1,des2,k=2)
        # print('match: ', matches)
        good = []; pts1 = []; pts2 = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance: good.append(m); pts1.append(kp1[m.queryIdx].pt); pts2.append(kp2[m.trainIdx].pt)
        
        # degengrade if not existing good matches
        # print('good len: ', len(good))
        if len(good)<min_matches:
            good = [];pts1 = [];pts2 = []
            for i,(m,n) in enumerate(matches):
                good.append(m); pts1.append(kp1[m.queryIdx].pt); pts2.append(kp2[m.trainIdx].pt)
        pts1 = np.array(pts1); PTS1.append(pts1);pts2 = np.array(pts2); PTS2.append(pts2)
    except:
        # if cannot find corresponding pairs, ignore this sift mask 
        PTS1.append([None]); PTS2.append([None])
    
    batch = 0
    pts1 = np.int32(np.round(PTS1[batch]))
    # print(pts1)
    # print(pts1.shape)
    coord1_flow_2D_norm_i = coord1_flow_2D[batch,:,pts1[:,1],pts1[:,0]].unsqueeze(0)
    coord2_flow_2D_norm_i = coord2_flow_2D[batch,:,pts1[:,1],pts1[:,0]].unsqueeze(0)

    out_of_range = (coord1_flow_2D_norm_i[:, 0, :] < 0) | (coord1_flow_2D_norm_i[:, 0, :] >= width) | \
                    (coord1_flow_2D_norm_i[:, 1, :] < 0) | (coord1_flow_2D_norm_i[:, 1, :] >= height) | \
                    (coord2_flow_2D_norm_i[:, 0, :] < 0) | (coord2_flow_2D_norm_i[:, 0, :] >= width) | \
                    (coord2_flow_2D_norm_i[:, 1, :] < 0) | (coord2_flow_2D_norm_i[:, 1, :] >= height)
    
    # Drop the points that are out of range
    coord1_flow_2D_norm_i = torch.masked_select(coord1_flow_2D_norm_i, ~out_of_range)
    coord2_flow_2D_norm_i = torch.masked_select(coord2_flow_2D_norm_i, ~out_of_range)

    coord1_flow_2D_norm_i = coord1_flow_2D_norm_i.reshape(1, 3, -1)
    coord2_flow_2D_norm_i = coord2_flow_2D_norm_i.reshape(1, 3, -1)

    points_images1 = coord1_flow_2D_norm_i[0, :2, :].cpu().numpy()
    points_images2 = coord2_flow_2D_norm_i[0, :2, :].cpu().numpy()

    al_err, coor_err, distances = compute_alignment_score(points_images1, points_images2)
    direct_npts = optimize_alierr(points_images1, points_images2, al_err, (height, width))

    #------------------------save points matching img---------------------------------------------
    points_match_path = os.path.join(save_door, "points_matchings_" + str(video_idx) +".png")
    dir = select_direct(al_err, direct_npts, points_images2)
    
    save_visualization(points_images1, points_images2, al_err, dir, cur_cv, tar_cv, points_match_path)
    
    print(f"best dir and kadfp error: {dir}, {al_err}")
    # return Points_match, send_text
    return  dir, al_err

def flow2drone(current_image, target_image, flow, video_idx, intrinsic, depth_c, depthfactor, save_door=""):
    b, _, h, w = flow.size()
    height, width,_ = current_image.shape

    coord1_flow_2D, coord2_flow_2D = flow2coord(flow)    # Bx3x(H*W) 
    coord1_flow_2D = coord1_flow_2D.view(b,3,h,w)        
    coord2_flow_2D = coord2_flow_2D.view(b,3,h,w)

    min_matches = 20

    PTS1=[]; PTS2=[];                                       # point list

    sift = cv2.SIFT_create()

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    cur_cv = current_image
    tar_cv = target_image

    # detect key points
    kp1, des1 = sift.detectAndCompute(cur_cv.astype(np.uint8),None)
    kp2, des2 = sift.detectAndCompute(tar_cv.astype(np.uint8),None)

    try:
        # filter out some key points
        matches = flann.knnMatch(des1,des2,k=2)
        # print('match: ', matches)
        good = []; pts1 = []; pts2 = []
        for i,(m,n) in enumerate(matches):
            
            if m.distance < 0.8*n.distance: good.append(m); pts1.append(kp1[m.queryIdx].pt); pts2.append(kp2[m.trainIdx].pt)
        
        if len(good)<min_matches:
            good = [];pts1 = [];pts2 = []
            for i,(m,n) in enumerate(matches):
                good.append(m); pts1.append(kp1[m.queryIdx].pt); pts2.append(kp2[m.trainIdx].pt)
        pts1 = np.array(pts1); PTS1.append(pts1);pts2 = np.array(pts2); PTS2.append(pts2)
    except:
        # if cannot find corresponding pairs, ignore this sift mask 
        PTS1.append([None]); PTS2.append([None])
    
    batch = 0
    pts1 = np.int32(np.round(PTS1[batch]))
    coord1_flow_2D_norm_i = coord1_flow_2D[batch,:,pts1[:,1],pts1[:,0]].unsqueeze(0)
    coord2_flow_2D_norm_i = coord2_flow_2D[batch,:,pts1[:,1],pts1[:,0]].unsqueeze(0)

    out_of_range = (coord1_flow_2D_norm_i[:, 0, :] < 0) | (coord1_flow_2D_norm_i[:, 0, :] >= width) | \
                    (coord1_flow_2D_norm_i[:, 1, :] < 0) | (coord1_flow_2D_norm_i[:, 1, :] >= height) | \
                    (coord2_flow_2D_norm_i[:, 0, :] < 0) | (coord2_flow_2D_norm_i[:, 0, :] >= width) | \
                    (coord2_flow_2D_norm_i[:, 1, :] < 0) | (coord2_flow_2D_norm_i[:, 1, :] >= height)
    
    # Drop the points that are out of range
    coord1_flow_2D_norm_i = torch.masked_select(coord1_flow_2D_norm_i, ~out_of_range)
    coord2_flow_2D_norm_i = torch.masked_select(coord2_flow_2D_norm_i, ~out_of_range)

    coord1_flow_2D_norm_i = coord1_flow_2D_norm_i.reshape(1, 3, -1)
    coord2_flow_2D_norm_i = coord2_flow_2D_norm_i.reshape(1, 3, -1)

    points_images1 = coord1_flow_2D_norm_i[0, :2, :].cpu().numpy()
    points_images2 = coord2_flow_2D_norm_i[0, :2, :].cpu().numpy()
    
    al_err, coor_err, distances = compute_alignment_score(points_images1, points_images2)
    # direct_npts = optimize_alierr(points_images1, points_images2, al_err, (height, width))
    dir = keypoints2dir(intrinsic, al_err, depth_c, depthfactor, points_images1.T, points_images2.T)
    
    #------------------------save points matching img---------------------------------------------
    points_match_path = os.path.join(save_door, "points_matchings_" + str(video_idx) +".png")

    save_visualization(points_images1, points_images2, al_err, dir, cur_cv, tar_cv, points_match_path)
    
    print(f"best dir and kadfp error: {dir}, {al_err}")
    
    return dir, al_err

def keypoints2dir(intrinsic, al_err, depth_c, depthfactor, points_images1, points_images2):
    """
    根据第一张图的相机内参 K、16位深度图 depth_c 及其尺度因子 depthfactor，
    以及在第一张图和第二张图之间匹配的特征点对 (points_images1, points_images2)，
    估计从第一张图到第二张图的相机平移大致方向和水平方向旋转角度。

    输入:
        intrinsic       : ndarray 相机内参
        depth_c         : ndarray，与第一张图同大小的16位深度图
        depthfactor     : float，将depth_c中的深度值转换为米的因子
        points_images1  : ndarray (N, 2)，第一张图的特征点(像素坐标)
        points_images2  : ndarray (N, 2)，第二张图的特征点(像素坐标)，与 points_images1 对应
    输出:
        drone_command   : str，无人机指令，格式为 "dx dy dz yaw"，其中 dx, dy, dz 为相机平移方向的速度，单位为 m/s；yaw 为水平方向旋转角度，单位为度
    """

    # 1) 获取 fx, fy, cx, cy
    fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]

    # 2) 构造 solvePnP 所需的 3D-2D 对应关系
    #    objectPoints: 来自第一张图，结合深度 -> 3D坐标
    #    imagePoints : 在第二张图中的 2D 像素坐标
    object_points = []
    image_points = []

    # 遍历特征点对
    for (u1, v1), (u2, v2) in zip(points_images1, points_images2):
        # 像素坐标通常需要先取整，否则索引会报错
        u1i = int(round(u1))
        v1i = int(round(v1))
        # print("cood:",u1i, v1i)
        # 从第一张图的深度图中取出该点深度值
        d = depth_c[v1i, u1i]
        # print("d:",d)
        # 若深度值无效或 0，跳过
        if d <= 0:
            continue
        # 转换为米
        Z = d / depthfactor
        # 根据 pinhole 模型，将 (u1, v1, Z) 转成相机坐标系下的 (X, Y, Z)
        X = (u1 - cx) / fx * Z
        Y = (v1 - cy) / fy * Z

        object_points.append([X, Y, Z])
        image_points.append([u2, v2])
    
    object_points = np.array(object_points, dtype=np.float32)
    image_points  = np.array(image_points,  dtype=np.float32)

    # 若有效的 3D-2D 点对太少，无法进行 solvePnP
    if len(object_points) < 4:
        print("Warning: Not enough valid 3D-2D correspondences.")
        return "original"

    # 3) 使用 OpenCV 的 solvePnP / solvePnPRansac 求解相对位姿
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        intrinsic,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        print("Warning: solvePnP failed to find a valid solution.")
        return "original"

    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(rvec)

    # 4) 提取“水平方向”旋转角度 (yaw)
    # 简化做法：取旋转后在相机系下的“前向轴” R[:,2]，并计算其在 x-z 平面的偏转角
    # 这里令 yaw = arctan2(forward_x, forward_z) (注意正负号对应左右)
    forward_vector = R[:, 2]          # 第三列即新的 z 轴在旧坐标系下的方向
    yaw_rad = math.atan2(forward_vector[0], forward_vector[2])
    yaw_deg = math.degrees(yaw_rad)/3   # 转换为度数；>0 => 向右，<0 => 向左
    if abs(yaw_deg)<0.5: # 旋轉角度閾值，低於該值則認為無需旋轉
        yaw_deg = 0.0
    if yaw_deg >4.0 and yaw_deg<180.0: # 限制在 [-4.0, 4.0] 之间
        yaw_deg = 4.0
    elif yaw_deg <-4.0 and yaw_deg>-180.0:
        yaw_deg = -4.0

    # 5) 提取平移向量，并判断主方向
    t = tvec.reshape(-1)  # shape (3,)
    # 这里假设相机坐标系: x→右, y→下, z→前
    # 先找到绝对值最大的分量，认为它是主要运动方向
    idx = np.argmax(np.abs(t))
    sign = np.sign(t[idx])

    # 6) 整理成无人机指令
    # v = round(al_err / 300, 3)  # 无人机速度
        
    # drone_command = ""
    dir = ""
    if idx == 0:
        # x方向 (sign>0 => right, sign<0 => left)
        if sign > 0:
            # drone_command = "0 "+str(v)+" 0"
            dir = "right"
        else:
            # drone_command = "0 -"+str(v)+" 0"
            dir = "left"
    elif idx == 1:
        # y方向 (sign>0 => down, sign<0 => up)
        if sign > 0:
            # drone_command = "0 0 -"+str(v)
            dir = "down"
        else:
            # drone_command = "0 0 "+str(v)
            dir = "up"
    else:
        # z方向 (sign>0 => forward, sign<0 => backward)
        if sign > 0:
            # drone_command = str(v)+" 0 0"
            dir = "backward"
        else:
            # drone_command = "-"+str(v)+" 0 0"
            dir = "forward"

    return  dir #, drone_command, str(round(-yaw_deg,3))


def kadfperr_for_dfae(curr_image, target_image, flow):
    b, _, h, w = flow.size()
    height, width,_ = curr_image.shape
    coord1_flow_2D, coord2_flow_2D = flow2coord(flow)    # Bx3x(H*W) 
    coord1_flow_2D = coord1_flow_2D.view(b,3,h,w)        
    coord2_flow_2D = coord2_flow_2D.view(b,3,h,w)
    min_matches = 20
    PTS1=[]; PTS2=[];                                       # point list
    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    cur_cv = curr_image
    tar_cv = target_image
    kp1, des1 = sift.detectAndCompute(cur_cv.astype(np.uint8),None)
    kp2, des2 = sift.detectAndCompute(tar_cv.astype(np.uint8),None)
    try:
        # filter out some key points
        matches = flann.knnMatch(des1,des2,k=2)
        good = []; pts1 = []; pts2 = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance: good.append(m); pts1.append(kp1[m.queryIdx].pt); pts2.append(kp2[m.trainIdx].pt)
        
        if len(good)<min_matches:
            good = [];pts1 = [];pts2 = []
            for i,(m,n) in enumerate(matches):
                good.append(m); pts1.append(kp1[m.queryIdx].pt); pts2.append(kp2[m.trainIdx].pt)
        pts1 = np.array(pts1); PTS1.append(pts1);pts2 = np.array(pts2); PTS2.append(pts2)
    except:
        # if cannot find corresponding pairs, ignore this sift mask 
        PTS1.append([None]); PTS2.append([None])
    batch = 0
    pts1 = np.int32(np.round(PTS1[batch]))
    # print(pts1)
    # print(pts1.shape)
    coord1_flow_2D_norm_i = coord1_flow_2D[batch,:,pts1[:,1],pts1[:,0]].unsqueeze(0)
    coord2_flow_2D_norm_i = coord2_flow_2D[batch,:,pts1[:,1],pts1[:,0]].unsqueeze(0)

    out_of_range = (coord1_flow_2D_norm_i[:, 0, :] < 0) | (coord1_flow_2D_norm_i[:, 0, :] >= width) | \
                    (coord1_flow_2D_norm_i[:, 1, :] < 0) | (coord1_flow_2D_norm_i[:, 1, :] >= height) | \
                    (coord2_flow_2D_norm_i[:, 0, :] < 0) | (coord2_flow_2D_norm_i[:, 0, :] >= width) | \
                    (coord2_flow_2D_norm_i[:, 1, :] < 0) | (coord2_flow_2D_norm_i[:, 1, :] >= height)
    
    # Drop the points that are out of range
    coord1_flow_2D_norm_i = torch.masked_select(coord1_flow_2D_norm_i, ~out_of_range)
    coord2_flow_2D_norm_i = torch.masked_select(coord2_flow_2D_norm_i, ~out_of_range)

    coord1_flow_2D_norm_i = coord1_flow_2D_norm_i.reshape(1, 3, -1)
    coord2_flow_2D_norm_i = coord2_flow_2D_norm_i.reshape(1, 3, -1)

    points_images1 = coord1_flow_2D_norm_i[0, :2, :].cpu().numpy()
    points_images2 = coord2_flow_2D_norm_i[0, :2, :].cpu().numpy()

    al_err, coor_err, distances = compute_alignment_score(points_images1, points_images2)
    return   al_err

def blended_images_for_dfae(target_image, curr_image, flow, tar_idx, cur_idx, save_door=""):
    b, _, h, w = flow.size()
    height, width,_ = curr_image.shape

    coord1_flow_2D, coord2_flow_2D = flow2coord(flow)    # Bx3x(H*W) 
    coord1_flow_2D = coord1_flow_2D.view(b,3,h,w)        
    coord2_flow_2D = coord2_flow_2D.view(b,3,h,w)
    min_matches = 20

    PTS1=[]; PTS2=[];                                       # point list
    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    cur_cv = curr_image
    tar_cv = target_image
    kp1, des1 = sift.detectAndCompute(cur_cv.astype(np.uint8),None)
    kp2, des2 = sift.detectAndCompute(tar_cv.astype(np.uint8),None)
    try:
        matches = flann.knnMatch(des1,des2,k=2)
        good = []; pts1 = []; pts2 = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance: good.append(m); pts1.append(kp1[m.queryIdx].pt); pts2.append(kp2[m.trainIdx].pt)
        if len(good)<min_matches:
            good = [];pts1 = [];pts2 = []
            for i,(m,n) in enumerate(matches):
                good.append(m); pts1.append(kp1[m.queryIdx].pt); pts2.append(kp2[m.trainIdx].pt)
        pts1 = np.array(pts1); PTS1.append(pts1);pts2 = np.array(pts2); PTS2.append(pts2)
    except:
        # if cannot find corresponding pairs, ignore this sift mask 
        PTS1.append([None]); PTS2.append([None])
    
    batch = 0
    pts1 = np.int32(np.round(PTS1[batch]))
    coord1_flow_2D_norm_i = coord1_flow_2D[batch,:,pts1[:,1],pts1[:,0]].unsqueeze(0)
    coord2_flow_2D_norm_i = coord2_flow_2D[batch,:,pts1[:,1],pts1[:,0]].unsqueeze(0)

    out_of_range = (coord1_flow_2D_norm_i[:, 0, :] < 0) | (coord1_flow_2D_norm_i[:, 0, :] >= width) | \
                    (coord1_flow_2D_norm_i[:, 1, :] < 0) | (coord1_flow_2D_norm_i[:, 1, :] >= height) | \
                    (coord2_flow_2D_norm_i[:, 0, :] < 0) | (coord2_flow_2D_norm_i[:, 0, :] >= width) | \
                    (coord2_flow_2D_norm_i[:, 1, :] < 0) | (coord2_flow_2D_norm_i[:, 1, :] >= height)
    
    # Drop the points that are out of range
    coord1_flow_2D_norm_i = torch.masked_select(coord1_flow_2D_norm_i, ~out_of_range)
    coord2_flow_2D_norm_i = torch.masked_select(coord2_flow_2D_norm_i, ~out_of_range)

    coord1_flow_2D_norm_i = coord1_flow_2D_norm_i.reshape(1, 3, -1)
    coord2_flow_2D_norm_i = coord2_flow_2D_norm_i.reshape(1, 3, -1)

    points_images1 = coord1_flow_2D_norm_i[0, :2, :].cpu().numpy()
    points_images2 = coord2_flow_2D_norm_i[0, :2, :].cpu().numpy()

    al_err, coor_err, distances = compute_alignment_score(points_images1, points_images2)
    direct_npts = optimize_alierr(points_images1, points_images2, al_err, (height, width))

    #------------------------save points matching img---------------------------------------------
    points_match_path = os.path.join(save_door, str(tar_idx)+"_"+str(cur_idx) +".png")

    save_visualization(points_images1, points_images2, al_err, "Keypoint Align Error ", cur_cv, tar_cv, points_match_path)
    # return Points_match, send_text
    return  al_err