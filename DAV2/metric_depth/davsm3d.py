from moviepy.editor import ImageSequenceClip
import argparse
import cv2
import numpy as np
import os
import torch
import logging
import socket
import select
from pathlib import Path
from collections import defaultdict
from distutils.log import info
import struct
import time
from depth_anything_v2.dpt import DepthAnythingV2

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--mode', type=str, default='da', choices=['da', 'm3d', 'eval'])
    parser.add_argument('--encoder_da', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--datasets_da', type=str, default='hypersim', choices=['hypersim', 'vkitti'],
                        help='[hypersim] for indoor, [vkitti] for outdoor, default: %(default)s')
    parser.add_argument('--encoder_m3d', type=str, default='small', choices=['small', 'large'])
    parser.add_argument('--max_depth', type=float, default=20)
    parser.add_argument('--input_path', type=Path, default='../datasets/tum_rgbd/')
    parser.add_argument('--save_root', type=Path, default='../depth_eval/')
    return parser.parse_args()

def undistort_image(image, intrinsic, distortion_coeffs):
    intrinsic_matrix = np.array([[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]])
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, intrinsic_matrix, distortion_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]
    return undistorted_image

def get_da_depth(img_name, raw_image):
    print(f'Progress {img_name}')
    depth = depth_anything.infer_image(raw_image, 518)
    # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    # depth = depth.astype(np.uint8)
    depth = depth * intrinsic[0] / 1000.0 # now the depth is metric
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65535
    depth[depth > 60000] = 0
    depth = depth.astype(np.uint16)
    return depth

def get_m3d_depth(metric3d_model, img_name, rgb, size, pad_info, intrinsic):
    print(f'Progress {img_name}')
    with torch.no_grad():
        pred_depth,_,_= metric3d_model.inference({'input': rgb})
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
        pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], size, mode='bilinear').squeeze()
        canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
        pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
        pred_depth = torch.clamp(pred_depth, 0, 300)
        depth = pred_depth.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65535
        depth[depth > 64000] = 0
        depth = depth.astype(np.uint16)
        return depth

def load_depth(path):
    # 读取16位无符号深度图
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # 获取深度图的中心区域
    h, w = depth.shape
    center_h, center_w = h // 2, w // 2
    half_h, half_w = center_h // 2, center_w // 2
    depth = depth[center_h - half_h:center_h + half_h, center_w - half_w:center_w + half_w]
    # depth是一个16位整型数组，范围[0, 65535]
    # 根据TUM深度标定，将其转为实际米（假设比例为 1/5000）
    depth_in_meters = depth.astype(np.float32) / 5000.0
    return depth_in_meters

def compute_metrics(gt, pred):
    # 过滤无效值（假设深度 > 0 为有效）
    mask = (gt > 0) & (pred > 0)
    gt_valid = gt[mask]
    pred_valid = pred[mask]
    
    # 计算误差
    diff = pred_valid - gt_valid
    abs_diff = np.abs(diff)
    
    # 基本误差项
    mae = np.mean(abs_diff)  # Mean Absolute Error
    rmse = np.sqrt(np.mean(diff**2))  # Root Mean Squared Error
    abs_rel = np.mean(abs_diff / gt_valid)
    sq_rel = np.mean((diff**2) / gt_valid)
    
    # 阈值精度指标
    # delta < 1.25, 1.25^2, 1.25^3
    def threshold_acc(gt_val, pr_val, threshold):
        ratio = np.maximum(gt_val / pr_val, pr_val / gt_val)
        return np.mean(ratio < threshold)
    
    delta1 = threshold_acc(gt_valid, pred_valid, 1.25)
    delta2 = threshold_acc(gt_valid, pred_valid, 1.25**2)
    delta3 = threshold_acc(gt_valid, pred_valid, 1.25**3)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'AbsRel': abs_rel,
        'SqRel': sq_rel,
        'Delta1': delta1,
        'Delta2': delta2,
        'Delta3': delta3
    }

def evaluate_networks(depth_gt_path, depth_da_path, depth_m3d_path):
    # 获取文件列表，
    metrics_da = {'MAE': [], 'RMSE': [], 'AbsRel': [], 'SqRel': [], 'Delta1': [], 'Delta2': [], 'Delta3': []}
    metrics_m3d = {'MAE': [], 'RMSE': [], 'AbsRel': [], 'SqRel': [], 'Delta1': [], 'Delta2': [], 'Delta3': []}
    imgs_num = len(depth_gt_path)
    img_idx = 0
    pbar = tqdm(total=imgs_num)
    while img_idx < imgs_num:
        gt_depth = load_depth(depth_gt_path[img_idx])
        da_depth = load_depth(depth_da_path[img_idx])
        m3d_depth = load_depth(depth_m3d_path[img_idx])
        # print(f'Progress {img_idx}, gt_depth: {depth_gt_path[img_idx]}, da_depth: {depth_da_path[img_idx]}, m3d_depth: {depth_m3d_path[img_idx]}')
        # 计算da指标
        res_net1 = compute_metrics(gt_depth, da_depth)
        for k in metrics_da.keys():
            metrics_da[k].append(res_net1[k])
        
        # 计算m3d指标
        res_net2 = compute_metrics(gt_depth, m3d_depth)
        for k in metrics_m3d.keys():
            metrics_m3d[k].append(res_net2[k])

        img_idx+=1
        pbar.update(1)
    pbar.close()
    # 汇总平均指标
    final_metrics_da = {k: np.mean(v) for k, v in metrics_da.items()}
    final_metrics_m3d = {k: np.mean(v) for k, v in metrics_m3d.items()}
    
    return final_metrics_da, final_metrics_m3d

if __name__ == '__main__':
    args = parse_args()

    input_path = args.input_path
    img_path = input_path / 'rgb/'
    gt_path = input_path / 'depth/'
    save_root = args.save_root
    save_root.mkdir(exist_ok = True, parents=True)
    output_path = save_root / input_path.name
    output_path.mkdir(exist_ok = True, parents=True)
    da_output_path = output_path / 'da/'
    da_output_path.mkdir(exist_ok = True, parents=True)
    m3d_output_path = output_path / 'm3d/'
    m3d_output_path.mkdir(exist_ok = True, parents=True)
    
    associate_list  = []
    depth_anything = None
    metric3d_model = None

    if args.mode == 'da':
        # DepthAnythingV2 initialize
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        depth_anything = DepthAnythingV2(**{**model_configs[args.encoder_da], 'max_depth': args.max_depth})
        depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{args.datasets}_{args.encoder_da}.pth', map_location='cpu'))
        depth_anything = depth_anything.to(DEVICE).eval()
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s | [%(levelname)s] | %(message)s')
        # DepthAnythingV2 initialize end
    elif args.mode == 'm3d':
        metric3d_model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_'+args.encoder_m3d, pretrain=True)
        metric3d_model.cuda().eval()
    
    # metric3d initialize
    # intrinsic = [517.417475, 585.870258, 336.769936, 207.547167]
    # intrinsic = [535.4, 539.2, 320.1, 247.6]
    intrinsic = [520.908620, 521.007327, 325.141442, 249.701764]
    distortion_coeffs = np.array([0.231222,-0.784899,-0.003257, -0.000105, 0.917205])

    h_origin, w_origin = 480, 640
    input_size = (616, 1064) # for vit model
    scale = min(input_size[0] / h_origin, input_size[1] / w_origin)
    padding = [123.675, 116.28, 103.53]
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    # metric3d initialize end

    if args.mode != 'eval':
        img_idx = 0
        img_names = sorted([f for f in os.listdir(img_path) if f.endswith('.png')], key=lambda x: (int(x.split('.')[0]),int(x.split('.')[1])))
        # 獲取完整的圖片文件路徑列表
        imgs = [os.path.join(img_path, img) for img in img_names]
        img_num = len(imgs)
        print(f'img_num: {img_num}')
        
        while img_idx<img_num:
            depth = None
            output_path = None
            curr_img = cv2.imread(imgs[img_idx])
            undistort_image(curr_img, intrinsic, distortion_coeffs)

            if args.mode=='da': # 执行DepthanythingV2得到深度图
                start_time1 = time.time()
                depth = get_da_depth(img_names[img_idx], curr_img) 
                cost1 = time.time() - start_time1
                print(f"get_depth執行時間：{cost1} 秒")
                output_path = os.path.join(da_output_path, img_names[img_idx])

            elif args.mode=='m3d': # 执行Metric3D得到深度图
                start_time1 = time.time()
                rgb = cv2.resize(curr_img, (int(w_origin * scale), int(h_origin * scale)), interpolation=cv2.INTER_LINEAR)
                intrinsic_scaled = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
                h, w = rgb.shape[:2]
                pad_h, pad_w = input_size[0] - h, input_size[1] - w
                pad_h_half, pad_w_half = pad_h // 2, pad_w // 2
                rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
                pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
                rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
                rgb = torch.div((rgb - mean), std)
                rgb = rgb[None, :, :, :].cuda()
                # 执行Metric3D得到深度图
                depth = get_m3d_depth(metric3d_model, img_names[img_idx], rgb, (h_origin,w_origin), pad_info, intrinsic_scaled)
                cost1 = time.time() - start_time1
                print(f"get_depth執行時間：{cost1} 秒") 
                output_path = os.path.join(m3d_output_path, img_names[img_idx])

            print(f'Save to {output_path}')
            cv2.imwrite(output_path, depth)
            img_idx += 1
    else:
        gt_names = sorted([f for f in os.listdir(gt_path) if f.endswith('.png')], key=lambda x: (int(x.split('.')[0]),int(x.split('.')[1])))
        depth_gt_path = [os.path.join(gt_path, f) for f in gt_names]
        pred_depth_names = sorted([f for f in os.listdir(da_output_path) if f.endswith('.png')], key=lambda x: (int(x.split('.')[0]),int(x.split('.')[1])))
        depth_da_path = [os.path.join(da_output_path, img) for img in pred_depth_names]
        depth_m3d_path = [os.path.join(m3d_output_path, img) for img in pred_depth_names]
        
        da_metrics, m3d_metrics = evaluate_networks(depth_gt_path, depth_da_path, depth_m3d_path)

        print("da metrics:")
        for k, v in da_metrics.items():
            print(f"{k}: {v:.4f}")
        
        print("m3d metrics:")
        for k, v in m3d_metrics.items():
            print(f"{k}: {v:.4f}")