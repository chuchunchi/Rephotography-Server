import argparse
import cv2U
import numpy as np
import logging

from pathlib import Path
from collections import defaultdict
from distutils.log import info

# KADFP import
from pathlib import Path
from pprint import pformat

import torch.nn.functional as F
from datetime import datetime
from skimage import io
import sys
import configargparse
import os
import torch

sys.path.append('../hloc')
from utils.read_write_model import read_model, qvec2rotmat
from utils.flow_estimator import Flow_estimator
from utils.flow_match_commend import kadfperr_for_dfae,flow2coord,save_visualization

import extract_features, match_features
import cv2
import numpy as np
import argparse
from fastdtw import fastdtw
import time
from moviepy.editor import ImageSequenceClip

def parse_args():
    parser = argparse.ArgumentParser(description='DTW')

    parser.add_argument('--target', '-t',type=str, default='target', 
                        help='target video name, default: %(default)s')
    parser.add_argument('--result', '-r', type=str, default='result',
                        help='result video name, default: %(default)s')
    parser.add_argument('--merge_idx', '-m', type=str, default='0,0')
    return parser.parse_args()

def get_twins_args():    
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument("--mixed_precision", type=str, default=True)
    parser.add_argument("--small", type=str, default=False)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--dim_corr", type=int, default=192)
    parser.add_argument("--dim_corr_coarse", type=int, default=64)
    parser.add_argument("--dim_corr_all", type=int, default=192)    
    parser.add_argument("--model", type=str, default="../hloc/pipelines/VBgps/pretrained_models/twins_one.pth")
    parser.add_argument("--fnet", type=str, default='twins')
    parser.add_argument("--twoscale", type=str, default=False)
    return parser.parse_known_args()[0]

def kadfp_distance(target_idx, result_idx):
    result_img = result_imgs[int(result_idx)]
    target_img = target_imgs[int(target_idx)]
    flow = estimator.estimate(result_img, target_img)
    error = kadfperr_for_dfae(result_img, target_img, flow)
    print(f"比较 target_idx: {int(target_idx)}, result_idx: {int(result_idx)} => error: {error}")

    tmp = [int(target_idx), int(result_idx), error]
    matched_error.append(tmp)
    return error

def extract_frames(video_path, start_idx):
    # 从视频中提取帧。
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames[start_idx:]

def save2video(imgs,save_name,fps=30):
    clip = ImageSequenceClip(imgs, fps)
    clip.write_videofile(save_name, codec='libx264')

def flow_to_point_match(curr_image, target_image, flow):
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
    return points_images1, points_images2


target_imgs = []
result_imgs = []
matched_error = []
if __name__ == '__main__':
    args = parse_args()

    # KADFP initialize
    print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
    print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['NN-superpoint']
    model_args = get_twins_args()
    logging.info('load RAFT model.')
    estimator = Flow_estimator(model_args)
    # KADFP initialize end
    
    target_name = str(args.target)
    result_name = str(args.result)
    target_video = '../../data/target/'+target_name+'/target.mp4'
    result_video = '../../data/outputs/'+result_name+'/result.mp4'
    print(f"target_video: {target_video}") 
    print(f"result_video: {result_video}")
    output_path = Path(f'../../data/outputs/{result_name}/DFAE/')
    output_path.mkdir(parents=True, exist_ok=True)
    frames_path = output_path / 'matched_frames/'
    frames_path.mkdir(parents=True, exist_ok=True)
    match_path = output_path / 'point_match/'
    match_path.mkdir(parents=True, exist_ok=True)
    target_merge_idx = int(args.merge_idx.split(',')[0])
    result_merge_idx = int(args.merge_idx.split(',')[1])

    target_imgs=extract_frames(target_video, target_merge_idx)
    result_imgs=extract_frames(result_video, result_merge_idx)

    target_idxs = np.arange(len(target_imgs))
    result_idxs = np.arange(len(result_imgs))
    print(len(target_idxs), len(result_idxs))

    matched_pairs_fix = []
    distance = 0

    with open(os.path.join(output_path,'matched_pairs.txt'), 'r') as f:
        matched_pairs_fix = f.readlines()
    distance = sum([float(line.split(' ')[2]) for line in matched_pairs_fix])

    dfae_mean = distance / len(matched_pairs_fix)
    errors = [float(line.split(' ')[2]) for line in matched_pairs_fix]
    median = np.median(errors)
    print(f"\n\n  target: {target_name}   result: {result_name}\n")
    print(f"  DFAE Total       : {distance:.5f}")
    print(f"  DFAE Median      : {median:.5f}")
    print(f"  DFAE frame-wise  : {dfae_mean:.5f}   (pixels)")
    

    min_err_pair,max_err_pair = [0,0,1000], [0,0,0]
    
    for line in matched_pairs_fix:
        data = line.split(' ')
        target_idx = int(data[0])
        result_idx = int(data[1])
        error = round(float(data[2]), 5)
        if error < min_err_pair[2]:
            min_err_pair = [target_idx, result_idx, error]
        if error > max_err_pair[2]:
            max_err_pair = [target_idx, result_idx, error]
        print(f"  target_idx: {target_idx}, result_idx: {result_idx}, error: {error}")
        pm1,pm2 = flow_to_point_match(target_imgs[target_idx], result_imgs[result_idx], estimator.estimate(target_imgs[target_idx], result_imgs[result_idx]))
        save_visualization(pm1,pm2, error, 'Align error:', target_imgs[target_idx], result_imgs[result_idx], os.path.join(match_path, f'{target_idx}_{result_idx}.png'))
    
    print(f"\n  min_err_pair: {min_err_pair}\n  max_err_pair: {max_err_pair}\n")
    
    frame_pairs = []
    for line in matched_pairs_fix:
        data = line.split(' ')
        target_idx = int(data[0])
        result_idx = int(data[1])
        error = round(float(data[2]), 5)
        frame_pairs.append([target_idx, result_idx, error])
    
    min_err_Tt_pairs = []
    for i in range(0,len(target_idxs)):
        local_min_err_pair = []
        local_min_err = 1000
        for frame_pair in frame_pairs:
            if frame_pair[0] == i:
                if frame_pair[2] < local_min_err and frame_pair[2] != 0:
                    local_min_err = frame_pair[2]
                    local_min_err_pair = frame_pair
        if local_min_err_pair != []:
            min_err_Tt_pairs.append(local_min_err_pair)
    
    Tt_error_sum = 0
    for min_err_pair in min_err_Tt_pairs:
        Tt_error_sum += min_err_pair[2]
        # print(min_err_pair)
    print(f"  DFAE Frame-wise Error         : {Tt_error_sum/len(target_idxs)}")

    min_err_tT_pairs = []
    for i in range(0,len(result_idxs)):
        local_min_err_pair = []
        local_min_err = 1000
        for frame_pair in frame_pairs:
            if frame_pair[1] == i:
                if frame_pair[2] < local_min_err and frame_pair[2] != 0:
                    local_min_err = frame_pair[2]
                    local_min_err_pair = frame_pair
        if local_min_err_pair != []:
            min_err_tT_pairs.append(local_min_err_pair)
    
    tT_error_sum = 0
    for min_err_pair in min_err_tT_pairs:
        tT_error_sum += min_err_pair[2]
        # print(min_err_pair)
    print(f"  DFAE Frame-wise inverse Error : {tT_error_sum/len(result_imgs)}")

    key_frame_idxs = []
    with open('../../data/outputs/'+ result_name +'/keyframe_idxs.txt', 'r') as f:
        key_frame_idxs_str = f.readlines()
        for idx in key_frame_idxs_str:
            if idx != '\n':
                key_frame_idxs.append(int(idx))
    f.close()
    
    min_err_key_pairs = []
    # key_pair_len = 0
    for i in key_frame_idxs:
        local_min_err_pair = []
        local_min_err = 1000
        for frame_pair in frame_pairs:
            if frame_pair[0] == i:
                # key_pair_len += 1
                if frame_pair[2] < local_min_err:
                    local_min_err = frame_pair[2]
                    local_min_err_pair = frame_pair
        if local_min_err_pair != []:
            min_err_key_pairs.append(local_min_err_pair)
    
    key_error_sum = 0
    for min_err_pair in min_err_key_pairs:
        # print(min_err_pair)
        key_error_sum += min_err_pair[2]
        
    print(f"  key pair average error: {key_error_sum/len(min_err_key_pairs)}\n\n")
    
    # 獲取文件夾內所有圖片文件名，並按名稱中的序號排序
    imgs = sorted([f for f in os.listdir(match_path) if f.endswith('.png')], key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0])))
    # 獲取完整的圖片文件路徑列表
    imgs = [os.path.join(match_path, img) for img in imgs]
    save2video(imgs, os.path.join(output_path, 'align_error.mp4'), fps=30)
