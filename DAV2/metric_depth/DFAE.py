import argparse
import cv2
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

sys.path.append('../hloc')
from utils.read_write_model import read_model, qvec2rotmat
from utils.flow_estimator import Flow_estimator
from utils.flow_match_commend import kadfperr_for_dfae, blended_images_for_dfae

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


target_imgs = []
result_imgs = []
matched_error = []
tar_idx,tes_idx = 0,0
output_path = ''
combined_frames_path = ''
matched_frames_path = ''
blended_frames_path = ''
if __name__ == '__main__':
    args = parse_args()
    
    target_name = str(args.target)
    result_name = str(args.result)
    target_video = '../../data/target/'+target_name+'/target.mp4'
    result_video = '../../data/outputs/'+result_name+'/result.mp4'
    print(f"target_video: {target_video}") 
    print(f"result_video: {result_video}")
    output_path = Path(f'../../data/outputs/{result_name}/DFAE/')
    output_path.mkdir(parents=True, exist_ok=True)
    matched_frames_path = output_path / 'matched_frames/'
    if matched_frames_path.exists():
        for file in matched_frames_path.glob('*'):
            file.unlink()
    matched_frames_path.mkdir(parents=True, exist_ok=True)

    combined_frames_path = output_path / 'combined_frames/'
    if combined_frames_path.exists():
        for file in combined_frames_path.glob('*'):
            file.unlink()
    combined_frames_path.mkdir(parents=True, exist_ok=True)

    blended_frames_path = output_path / 'blended_frames/'
    if blended_frames_path.exists():
        for file in blended_frames_path.glob('*'):
            file.unlink()
    blended_frames_path.mkdir(parents=True, exist_ok=True)
    target_merge_idx = int(args.merge_idx.split(',')[0])
    result_merge_idx = int(args.merge_idx.split(',')[1])

    target_imgs=extract_frames(target_video, target_merge_idx)
    result_imgs=extract_frames(result_video, result_merge_idx)

    target_idxs = np.arange(len(target_imgs))
    result_idxs = np.arange(len(result_imgs))
    print(len(target_idxs), len(result_idxs))

    matched_pairs_fix = []
    distance = 0

     # KADFP initialize
    print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
    print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['NN-superpoint']
    model_args = get_twins_args()
    logging.info('load RAFT model.')
    estimator = Flow_estimator(model_args)
    # KADFP initialize end

    if not os.path.exists(os.path.join(output_path,'matched_pairs.txt')):
        # 使用 fastdtw 计算 DTW 距离
        start_time = time.time()
        distance, dtw_path = fastdtw(target_idxs, result_idxs, dist=kadfp_distance)
        end_time = time.time() - start_time
        print(f"cost时间: {end_time:.5f} 秒")
        
        matched_pairs = []
        for target_idx, result_idx in dtw_path:
            for matched_tar_idx, matched_res_idx, error in matched_error:
                if(target_idx == matched_tar_idx and result_idx == matched_res_idx):
                    matched_pairs.append(f"{target_idx} {result_idx} {error}")
        
        # 通过将列表转换为集合然后再转换回列表来删除重复行
        matched_pairs_fix = list(set(matched_pairs))
        
        matched_pairs_fix.sort(key=lambda x: (int(x.split(' ')[0]), int(x.split(' ')[1])))
        with open(os.path.join(output_path,'matched_pairs.txt'), 'w') as f:
            for line in matched_pairs_fix:
                f.write(f"{line}\n")
        f.close()

    else:
        with open(os.path.join(output_path,'matched_pairs.txt'), 'r') as f:
            matched_pairs = f.readlines()
            merge_pos = 0
            for idx,line in enumerate(matched_pairs):
                if(line.split(' ')[0] == str(target_merge_idx) and line.split(' ')[1] == str(result_merge_idx)):
                    merge_pos = idx
                    break
            matched_pairs_fix = matched_pairs[merge_pos:]
        distance = sum([float(line.split(' ')[2]) for line in matched_pairs_fix])

    
    dfae_mean = distance / len(matched_pairs_fix)
    errors = [float(line.split(' ')[2]) for line in matched_pairs_fix]
    median = np.median(errors)
    print(f"\n\n  target: {target_name}   result: {result_name}\n")
    print(f"  DFAE Total       : {distance:.5f}")
    print(f"  DFAE Median      : {median:.5f}")
    print(f"  DFAE Mean        : {dfae_mean:.5f}   (pixels)")
    

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

        # 拼接帧
        combined_frame = np.hstack((target_imgs[target_idx], result_imgs[result_idx]))

        # 在左上角添加文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.imwrite(os.path.join(combined_frames_path, f'{target_idx}_{result_idx}_{error:.5f}.png'), combined_frame)
        # cv2.putText(combined_frame, f'target({target_idx}), result({result_idx}) -> distance: {str(error)}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(matched_frames_path, f'{target_idx}_{result_idx}.png'), combined_frame)
        
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
    print(f"  DFAE Frame-wise Error         : {Tt_error_sum/len(target_imgs)}")

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
        flow = estimator.estimate(target_imgs[min_err_pair[0]], result_imgs[min_err_pair[1]])
        blended_images_for_dfae(target_imgs[min_err_pair[0]], result_imgs[min_err_pair[1]], flow, min_err_pair[0], min_err_pair[1], blended_frames_path)

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
    imgs = sorted([f for f in os.listdir(matched_frames_path) if f.endswith('.png')], key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0])))
    # 獲取完整的圖片文件路徑列表
    imgs = [os.path.join(matched_frames_path, img) for img in imgs]
    save2video(imgs, os.path.join(output_path, 'DFAE.mp4'), fps=30)

    Tt_img_paths = []
    for pair in min_err_Tt_pairs:
        Tt_img_paths.append(os.path.join(matched_frames_path, f'{pair[0]}_{pair[1]}.png'))
    save2video(Tt_img_paths, os.path.join(output_path, 'DFAE_Fw.mp4'), fps=30)

    tT_img_paths = []
    for pair in min_err_tT_pairs:
        tT_img_paths.append(os.path.join(matched_frames_path, f'{pair[0]}_{pair[1]}.png'))
    save2video(tT_img_paths, os.path.join(output_path, 'combined.mp4'), fps=30)

    belended_imgs = []
    for pair in min_err_tT_pairs:
        belended_imgs.append(os.path.join(blended_frames_path, f'{pair[0]}_{pair[1]}.png'))
    save2video(belended_imgs, os.path.join(output_path, 'blended_error.mp4'), fps=30)
    
    # blended_imgs = sorted([f for f in os.listdir(blended_frames_path) if f.endswith('.png')], key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0])))
    # blended_imgs = [os.path.join(blended_frames_path, img) for img in blended_imgs]
    # save2video(blended_imgs, os.path.join(output_path, 'blended_error.mp4'), fps=30)

    blended_frames_path