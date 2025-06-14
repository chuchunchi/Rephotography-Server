import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import logging
import socket
import select
from pathlib import Path
from collections import defaultdict
from distutils.log import info

from depth_anything_v2.dpt import DepthAnythingV2

# KADFP import
from pathlib import Path
from pprint import pformat
import time
import h5py
import torch.nn.functional as F
from datetime import datetime
from skimage import io
import sys
import configargparse
import os
import random
import flow_vis

sys.path.append('../hloc')
import extractors
from utils.read_write_model import read_model, qvec2rotmat
from utils.io import list_h5_names
from utils.base_model import dynamic_load
from utils.flow_estimator import Flow_estimator
from utils.flow_match_commend import flow_to_drone

import extract_features, match_features
import pairs_from_covisibility, pairs_from_retrieval
import colmap_from_nvm, triangulation, localize_sfm


def parse_args():
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    # parser.add_argument('--img-path', type=str)
    parser.add_argument('--video_only', action='store_true', help='only estimate the pose of the video frames')
    parser.add_argument('--input-size', type=int, default=518)
    # parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    # parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    # parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--dataset', type=Path, default='../datasets/1206_drone',
                    help='Path to the dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='../outputs/1206_drone',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--target_video', type=Path, default='../datasets/testvideo/test2.mp4',
                        help='Path to the target image, default: %(default)s')
    parser.add_argument('--num_loc', type=int, default=20,
                        help='Number of image pairs for loc, default: %(default)s')
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
    # parser.add_argument('--model', type=str, default="./snapshot/Twins_train_with_pretrain_smaple_len_250/checkpoint_latest.pth")  
    parser.add_argument("--fnet", type=str, default='twins')
    parser.add_argument("--twoscale", type=str, default=False)
    return parser.parse_known_args()[0]

def get_depth(args, img_path, raw_image):
    print(f'Progress {img_path}')
    output_name = os.path.splitext(os.path.basename(img_path))[0]

    depth = depth_anything.infer_image(raw_image, args.input_size)
    # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    # depth = depth.astype(np.uint8)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65535.0

    output_path = os.path.join(os.path.dirname(img_path), "{}_disp.png".format(output_name))
    cv2.imwrite(output_path, depth.astype(np.uint16))
    return output_path

def load_db_gfeats(path, key='global_descriptor'):
    db_names = list_h5_names(path)
    with h5py.File(str(path), 'r') as fd:
        desc = [fd[n][key].__array__() for n in db_names]
    return desc, db_names

def load_db_localfeats(path):
    path = [path]
    name2ref = {n: i for i, p in enumerate(path)
                for n in list_h5_names(p)}
    return name2ref

def load_netvlad(retrieval_config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(extractors, retrieval_config['model']['name'])
    model = Model(retrieval_conf['model']).eval().to(device)
    return model

def parse_intrinsic(path):
    with open(path, 'r') as f:
        for line in f:
            line.strip('\n')
            data = line.split(' ')
            camera_model, width, height, *params = data
            params = np.array(params, float)
            info = (camera_model, int(width), int(height), params)
    return info

def pose_postproc(qvec ,tvec):
    R_t = np.eye(4)
    R = qvec2rotmat(qvec)
    R_t[:3, :3] = R.T
    R_t[:3, 3] = -R.T @ tvec
    result = np.array(R_t).reshape(-1,1)
    result = np.array(['{:+.11f}'.format(result[n][0]) for n in range(len(result))], dtype='a11').reshape(-1, 1)
    result = np.array2string(result)
    return result

def recv_info(sock, count):
    data = b''
    while count:
        # recv(count) arg 'count' is maximum size 
        buf = sock.recv(count)
        if not buf : return None
        data += buf
        count -= len(buf)
    return data

def send_data(sock, image_data=None, text_data=None):
    # 标志位
    flags = 0
    if image_data:
        flags |= 1  # 第0位
    if text_data:
        flags |= 2  # 第1位

    # 图片和字符串的长度
    image_length = len(image_data) if image_data else 0
    text_length = len(text_data.encode('utf-8')) if text_data else 0

    # 打包报头
    header = struct.pack('!BII', flags, image_length, text_length)

    # 发送报头
    conn.sendall(header)

    # 发送图片数据
    if image_data:
        conn.sendall(image_data)

    # 发送字符串数据
    if text_data:
        conn.sendall(text_data.encode('utf-8'))



if __name__ == '__main__':
    args = parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    save_path = 'assets/tmp/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | [%(levelname)s] | %(message)s')

    # KADFP initialize
    if not args.video_only:
        dataset = args.dataset
        images = dataset / 'images_upright/'
        images.mkdir(exist_ok = True, parents=True)

        Stage_2_results_path = dataset / 'xr/'
        Stage_2_results_path.mkdir(exist_ok = True, parents=True)

        target_video = args.target_video

        outputs = args.outputs  # where everything will be saved
        reference_sfm = outputs / 'sfm_superpoint+NN'  # the SfM model we will build
        db_global_feats = outputs / f'global-feats-netvlad.h5'
        db_feats = outputs / f'feats-superpoint-n4096-r1024.h5'
        loc_pairs = outputs / f'pairs-query-netvlad{args.num_loc}.txt'  # top-k retrieved by NetVLAD
        results = outputs / f'Aachen_hloc_superpoint+superglue_netvlad{args.num_loc}.txt'

        # list the standard configurations available
        print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
        print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

        # pick one of the configurations for extraction and matching
        retrieval_conf = extract_features.confs['netvlad']
        feature_conf = extract_features.confs['superpoint_aachen']
        matcher_conf = match_features.confs['NN-superpoint']
        intrinsic = dataset / 'queries' / 'intrinsic.txt'

        test_commend = [
        "up 100.000",
        "down 200.000",
        "left 300.000",
        "right 400.000",
        "forward 500.000",
        "back 600.000"
        ]

        model_args = get_twins_args()
        logging.info('load RAFT model.')
        estimator = Flow_estimator(model_args)
        logging.info('load SfM model.')
        sfm_model = read_model(str(reference_sfm))
        logging.info('load db global feats.')
        db_gdesc, db_names = load_db_gfeats(db_global_feats)
        logging.info('load NetVlad model.')
        netvlad = load_netvlad(retrieval_conf)
        logging.info('load db local feats.')
        db_local_name2ref = load_db_localfeats(db_feats)
        query_info = parse_intrinsic(intrinsic)
        cap = cv2.VideoCapture(str(target_video))
        frame_count = 0
    # KADFP initialize end

    # socket initialize
    ssock = socket.socket()
    addr = ('140.113.195.240', 9999)
    ssock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ssock.setblocking(False)
    ssock.bind(addr)
    ssock.listen(5)
    fds = [ssock]
    logs = defaultdict(lambda: defaultdict(list))
    logging.info("--------server is ready--------")
    
    while(True):
        # 删除旧数据
        for old_data in os.listdir(Path(save_path)):
            os.unlink(Path(save_path) / old_data)
        if not args.video_only:
            for old_img_name in os.listdir(images):
                os.unlink(images / old_img_name)
            for old_item in os.listdir(outputs):
                if '.txt' in old_item:
                    if old_item == 'pairs-exhaustive.txt':
                        pass
                    else:
                        os.unlink(outputs / old_item)
                elif '.7_pairs-query-netvlad20' in old_item:
                    os.unlink(outputs / old_item)
        # ==========================================

        logging.info("waiting for connection...")
        readsocks, _, _  = select.select(fds, [], [])
        for sock in readsocks:
            if sock is ssock:
                csock, addr = sock.accept()
                fds.append(csock)
                logging.info(f"got new connection from {addr}")
            else:
                c = sock.fileno() 
                logging.info(f"got query from {c}")
                # 接收图片序号
                frame_idx = recv_info(sock, 10)
                # print(f"frame idx = {frame_idx}<==========================")
                if frame_idx is not None:
                    print(frame_idx)
                    frame_idx = int(frame_idx.decode())
                    _ = recv_info(sock, 1)
                    logging.info(f"frame index : {frame_idx}")
                    # 接收图片size
                    img_size = recv_info(sock, 10).decode()
                    _ = recv_info(sock, 1)
                    logging.info(f"image size : {img_size}")
                    # 接收图片
                    query = recv_info(sock, int(img_size))
                    query = np.fromstring(query, np.uint8)
                    query_img = cv2.imdecode(query, cv2.IMREAD_COLOR)

                    query_img = cv2.resize(query_img,(640,480))
                    img_name = f"q_{frame_idx}.png"
                    img_path = Path(save_path, img_name)
                    cv2.imwrite(str(img_path), query_img)
                    
                    if query_img is not None:
                        # 执行Depth-Anything得到深度图
                        depth = get_depth(args, str(img_path), query_img)
                        output_img_bytes = cv2.imencode('.png', cv2.imread(depth, cv2.IMREAD_UNCHANGED))[1].tobytes()
                        img_size = len(output_img_bytes)
                        # print(f"img_size = {str(img_size).zfill(10)}")

                        # ================KADFP==================
                        if not args.video_only:
                            target_img = None
                            tmp_count = 0
                            while True:
                                ret, target_img = cap.read()
                                tmp_count += 1
                                if not ret or tmp_count % 5 == 0:
                                    break
                            if target_img is not None:
                                cv2.imwrite(str(img_path), target_img)
                                current_image_bgr = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
                                # Save the image
                                # Define the output file path
                                # output_path = os.path.join(Stage_2_results_path, "current" + str(frame_count) + ".jpg")
                                # cv2.imwrite(output_path, current_image_bgr)

                                # target_img = io.imread(target_video)
                                
                                start_time = time.time()
                                flow = estimator.estimate(query_img, target_img)
                                end_time  = time.time()
                                execution_time = end_time - start_time
                                print(f"flow estimate 執行時間：{execution_time} 秒") 
                                
                                Points_match, drone_commend = flow_to_drone(query_img, target_img, flow, Stage_2_results_path, frame_count)
                                frame_count += 1
                                #---------visualize flow ------------------------------------------
                                flow_np = flow.squeeze().detach().cpu().numpy()
                                flow_permuted = np.transpose(flow_np, (1, 2, 0))      
                                flow_visualized =  flow_vis.flow_to_color(flow_permuted, convert_to_bgr=True)
                                cv2.imwrite(os.path.join(Stage_2_results_path, "flow_visualized.png"), flow_visualized)
                                query_img_bgr = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
                                # sock.sendall(int(Stage).to_bytes(4, 'big'))
                                drone_commend_length = len(drone_commend)
                                print('drone_commend_length: ', drone_commend_length)
                                drone_commend_length_bytes = drone_commend_length.to_bytes(4, 'big')
                                print('drone_commend: ', drone_commend)
                                
                        
                        #发送图片到client
                        sock.sendall(('img '+ str(img_size).zfill(10)).encode())
                        sock.sendall(output_img_bytes)
                        if not args.video_only:
                            sock.sendall(('str'+ drone_commend_length_bytes).encode())
                            sock.sendall(drone_commend.encode())
                        
                    else:
                        print("empty image.")
                        # _ = recv_info(sock, 1)
                        continue
                else:
                    logging.info(f"connection {c} ends")
                    fds.remove(sock)
                    sock.close()
    cap.release()