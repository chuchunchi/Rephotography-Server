import os
from skimage import io
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import warnings
from tqdm import tqdm
import datetime
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys

# our model
# from flow_estimator import Flow_estimator
# from config import get_twins_args, get_eval_args, get_life_args

torch.set_grad_enabled(False)
import flow_vis
import time
# from colormap import get_colormap
# from eval_utils import resize_lighting, resize_viewpoint
# from metrics import metrics

import configargparse
import argparse

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

def save_visualization(coord1, coord2, al_err, coor_err, ref_cv, tar_cv, output_path=None):
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

    # 讀取參考圖像
    ref_bgr = cv2.cvtColor(ref_cv, cv2.COLOR_RGB2BGR)
    tar_bgr = cv2.cvtColor(tar_cv, cv2.COLOR_RGB2BGR)

     # 確保兩張照片的尺寸相同
    height, width, _ = ref_bgr.shape

    tar_bgr = cv2.resize(tar_bgr, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 0, 0)  # Green color in BGR format
    thickness = 2

    # 將兩張照片疊加在一起
    blended_image = cv2.addWeighted(ref_bgr, 0.5, tar_bgr, 0.5, 0)
    cv2.putText(blended_image, 'Al error: ' + str(al_err) , (10, 50), font, font_scale, color, thickness)
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
    # if output_path is not None:
        # cv2.imwrite(output_path, blended_image)

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

def select_direct(al_err, direct_npts, points_images2, blended_image, save_door=None):
    direct = 'original'
    best_points = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 0, 0)  # Green color in BGR format
    thickness = 2
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

def flow2error(current_image, target_image, flow):
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
            if m.distance < 0.8*n.distance: 
                good.append(m); pts1.append(kp1[m.queryIdx].pt); pts2.append(kp2[m.trainIdx].pt)

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
    return al_err

def flow_to_drone(current_image, target_image, flow, stage2_count, K, depth_c, depth_t, save_door=""):
    b, _, h, w = flow.size()
    height, width,_ = current_image.shape

    coord1_flow_2D, coord2_flow_2D = flow2coord(flow)    # Bx3x(H*W) 
    coord1_flow_2D = coord1_flow_2D.view(b,3,h,w)        
    coord2_flow_2D = coord2_flow_2D.view(b,3,h,w)

    margin = 10                 # avoid corner case
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

    # image1_with_keypoints = draw_keypoints(cur_cv, kp1)
    # image2_with_keypoints = draw_keypoints(tar_cv, kp2)
    # if save_door is not None:
        # cv2.imwrite(os.path.join(save_door, "current_image_with_keypoints.png"), image1_with_keypoints)
        # cv2.imwrite(os.path.join(save_door, "target_image_with_keypoints.png"), image2_with_keypoints)
    
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
    
    # Save points_images1 coordinates to fp1.txt
    # np.savetxt(os.path.join(save_door, str(stage2_count)+"_curr_fp.txt"), points_images1.T, fmt='%f')
    # np.savetxt(os.path.join(save_door, str(stage2_count)+"_vid_fp.txt"), points_images2.T, fmt='%f')

    al_err, coor_err, distances = compute_alignment_score(points_images1, points_images2)
    direct_npts = optimize_alierr(points_images1, points_images2, al_err, (height, width))

    #------------------------save points matching img---------------------------------------------
    # points_match_path = save_door / 'points_match_output/'
    # points_match_path.mkdir(exist_ok = True, parents=True)
    points_match_path = os.path.join(save_door, "points_matchings_" + str(stage2_count) +".png")

    Points_match = save_visualization(points_images1, points_images2, al_err, coor_err, cur_cv, tar_cv, points_match_path)

    direct = select_direct(al_err, direct_npts, points_images2, Points_match, save_door)
    # visual_flow_matching(coord1_flow_2D_norm_i, coord2_flow_2D_norm_i, cur_cv, tar_cv, save_door)
    print('direct:', direct)
    # if al_err <= 10.0:
    #     send_text = "original 0.0"
    # else:
    #     send_text = direct + ' ' + str(al_err)

    # if save_door!="":
    #     cv2.putText(Points_match, send_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    #     cv2.imwrite(points_match_path, Points_match)

    # Points_match = cv2.cvtColor(Points_match, cv2.COLOR_BGR2RGB)
    trans_command, rot_command = calculate_camera_motion(K, depth_c, depth_t, points_images1.T, points_images2.T)
    
    print(f"Move: {trans_command}, {rot_command}")
    
    if save_door!="":
        cv2.putText(Points_match, trans_command, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(Points_match, rot_command, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imwrite(points_match_path, Points_match)
    
    # return Points_match, send_text
    return Points_match, trans_command, rot_command

def points_2d_to_3d(points, depth, intrinsic_matrix):
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        
        points_3d = []
        for (u, v) in points:
            z = depth[int(v), int(u)] / 4000.0  # 假设深度单位为毫米，转换为米
            if z > 0:  # 忽略无效深度
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points_3d.append([x, y, z])
        return np.array(points_3d)

def fp2pose(fp1,fp2,K):
    try:
        E, mask = cv2.findEssentialMat(fp1, fp2, K)

        # 從本質矩陣恢復相機運動
        _, R, t, mask = cv2.recoverPose(E, fp1, fp2, K)
        return R, t
    except:
        return None, None


def calculate_camera_motion(K, depth1, depth2, fp1, fp2):
    # Convert 2D feature points to 3D points in camera coordinate
    points_3d_1 = points_2d_to_3d(fp1, depth1, K)
    points_3d_2 = points_2d_to_3d(fp2, depth2, K)
    # Find relative transformation using Procrustes analysis (least-squares fit)
    if len(points_3d_1) < 3 or len(points_3d_2) < 3:
        raise ValueError("Not enough 3D points for motion estimation.")

    # Match 3D points (assume fp1[i] corresponds to fp2[i])
    centroid_1 = np.mean(points_3d_1, axis=0)
    centroid_2 = np.mean(points_3d_2, axis=0)

    points_3d_1_centered = points_3d_1 - centroid_1
    points_3d_2_centered = points_3d_2 - centroid_2

    H = np.dot(points_3d_1_centered.T, points_3d_2_centered)
    U, S, Vt = np.linalg.svd(H)
    R_mat = np.dot(Vt.T, U.T)
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = np.dot(Vt.T, U.T)

    t_vec = (centroid_2 - np.dot(R_mat, centroid_1))/5

    # Keep the variable with the maximum absolute value and set the others to zero
    max_abs_value = max(abs(t_vec[0]), abs(t_vec[1]), abs(t_vec[2]))
    for i in range(3):
        if abs(t_vec[i]) != max_abs_value:
            t_vec[i] = 0.0
    
    # Analyze translation
    directions = {"forward": t_vec[2], "backward": -t_vec[2],
                  "left": -t_vec[0], "right": t_vec[0],
                  "up": t_vec[1], "down": -t_vec[1]}

    move_direction = max(directions, key=directions.get)
    move_distance = np.linalg.norm(t_vec)

    # Analyze rotation (only yaw considered)
    euler_angles = R.from_matrix(R_mat).as_euler('xyz', degrees=True)
    yaw_angle = euler_angles[1]/10 # yaw

    # if move_distance < 0.05:  # Threshold for negligible movement
    #     move_direction = "original"
    #     move_distance = 0.0
    # if move_distance > 1.0: # 限速
    #     move_distance = 1.0
    # if abs(yaw_angle) > 10.0:# 限制旋轉角度
    #     yaw_angle = 10.0
    print(f"dir:{round(t_vec[2],4)}, {round(t_vec[0],4)}, {round(t_vec[1],4)}, rot:{yaw_angle}")

    for i in range(3):
        if abs(t_vec[i]) < 0.005:
            t_vec[i] = 0.0
        elif abs(t_vec[i]) > 0.4:
            t_vec[i] = 0.4 if t_vec[i] > 0 else -0.4 
    
    if abs(yaw_angle) > 5.0:
        yaw_angle = 5.0 if yaw_angle > 0 else -5.0
    elif abs(yaw_angle) < 2.0:
        yaw_angle = 0.0

    # return move_direction+" " +str(round(move_distance,5)), str(round(yaw_angle,5))
    return str(round(t_vec[2],4))+ " "+ str(round(t_vec[0],4))+" "+str(round(t_vec[1],4)), str(round(yaw_angle,4))
