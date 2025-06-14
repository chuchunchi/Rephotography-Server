from moviepy.editor import ImageSequenceClip
import argparse
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
# import flow_vis


sys.path.append('../hloc')
import extractors
from utils.read_write_model import read_model, qvec2rotmat
# from utils.io import list_h5_names
# from utils.base_model import dynamic_load
from utils.flow_estimator import Flow_estimator
from utils.flow_match_commend import flow2drone

import extract_features, match_features
import traceback
import cv2
# import pairs_from_covisibility, pairs_from_retrieval
# import colmap_from_nvm, triangulation, localize_sfm


def parse_args():
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    # parser.add_argument('--img-path', type=str)
    parser.add_argument('--kadfp', action='store_true', help='activate KADFP model')
    parser.add_argument('--input-size', type=int, default=518)
    # parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--datasets', type=str, default='hypersim', choices=['hypersim', 'vkitti'],
                        help='[hypersim] for indoor, [vkitti] for outdoor, default: %(default)s')
    parser.add_argument('--max-depth', type=float, default=20)
    parser.add_argument('--outputs', type=Path, default='../outputs/', 
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--target_video', type=Path, default='../datasets/testvideo/test2.mp4',
                        help='Path to the target image, default: %(default)s')
    # parser.add_argument('--num_loc', type=int, default=20,
    #                     help='Number of image pairs for loc, default: %(default)s')
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
    depth = depth_anything.infer_image(raw_image, args.input_size)
    # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    # depth = depth.astype(np.uint8)
    # depth = depth * intrinsic[0] / 1000.0 # now the depth is metric
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65535
    # depth[depth > 50000] = 0
    depth = depth.astype(np.uint16)
    output_name = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(disp_path, "{}_disp.png".format(output_name))
    cv2.imwrite(output_path, depth)
    return depth

def save2video(img_path,save_name,sort_basis,fps=10):
    if len(os.listdir(img_path)) > 0:
        # 獲取文件夾內所有圖片文件名，並按名稱中的序號排序
        imgs = sorted([f for f in os.listdir(img_path) if f.endswith('.png')], key=lambda x: int(x.split('.')[0][sort_basis:]))
        imgs = [os.path.join(img_path, img) for img in imgs]
        clip = ImageSequenceClip(imgs, fps)
        clip.write_videofile(f"../datasets/testvideo/{save_name}", codec='libx264')
    else:
        print(f"No images in {img_path} !")


class Server:
    def __init__(self, host='', port=9999):
        self.host = host
        self.port = port
        self.sock = None
        self.conn = None
        self.addr = None
        self.curr_idx = 0
        self.video_idx = 0
        self.compare_counter = 0
        self.video_over = False

    def reset(self):
        self.sock = None
        self.conn = None
        self.addr = None
        self.curr_idx = 0
        self.video_idx = 0
        self.compare_counter = 0
        self.video_over = False

    def start(self):
        while True:
            try:
                self.reset()
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.bind((self.host, self.port))
                self.sock.listen(1)
                print('服务端正在监听端口', self.port)
                self.conn, self.addr = self.sock.accept()
                print('已连接客户端', self.addr)
                self.handle_client()
            except KeyboardInterrupt:
                # 允许通过 Ctrl+C 停止服务端
                print("服务器关闭")
                if args.kadfp:
                    save2video(str(output/"curr_frames/"), "test"+str(output).split('/')[2]+".mp4", 1, 10)
                    save2video(str(output/"align_error/"), "align_error"+str(output).split('/')[2]+".mp4", 17, 10)
                    keyframe_idxs = sorted(set(keyframe_idxs))
                    with open(str(output/"keyframe_idxs.txt"), 'w') as f:
                        for item in keyframe_idxs:
                            f.write(f"{item}\n")
                else:
                    save2video(str(output/"curr_frames/"), "target"+str(output).split('/')[2]+".mp4", 1, 10)
                time.sleep(1)
                break
            except Exception as e:
                # 捕获其他未知异常，确保服务器不会崩溃
                traceback.print_exc()
                time.sleep(1)
                continue

    def handle_client(self):
        loop_count = 0
        # 删除旧数据
        self.del_old_data()

        while True:
            loop_count += 1
            try:
                # 接收图像大小
                size_data = self.recvall(4)
                if not size_data:
                    break
                image_size = struct.unpack('!I', size_data)[0]
                # 接收图像序号
                idx_data = self.recvall(4)
                if not idx_data:
                    break
                image_idx = struct.unpack('!I', idx_data)[0]
                # 接收图像数据
                image_data = self.recvall(image_size)
                if not image_data:
                    break
                print('收到图像，大小为', image_size)
                nparr = np.frombuffer(image_data, np.uint8)
                curr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                curr_img = cv2.resize(curr_img,(640,480))
                self.curr_idx = image_idx
                img_name = f"q{image_idx}.png"
                
                img_path = str(curr_frame_path / img_name)
                cv2.imwrite(img_path, curr_img)
                # 执行DepthanythingV2得到深度图
                start_time1 = time.time()
                curr_depth = get_depth(args, img_path, curr_img) 
                cost1 = time.time() - start_time1
                print(f"get_depth執行時間：{cost1} 秒") 
                # 发送深度图
                self.send_depth(curr_depth,image_idx)
                
                if args.kadfp:
                    # 接收video_idx
                    v_idx_data = self.recvall(4)
                    if not v_idx_data:
                        break
                    self.video_idx = struct.unpack('!I', v_idx_data)[0]
                    print(f"接收到video_idx: {self.video_idx}")
                    keyframe_idxs.append(self.video_idx)
                    cap.set(cv2.CAP_PROP_POS_FRAMES,self.video_idx)
                    ret, img = cap.read()
                    target_img = cv2.resize(img,(640,480))
                    
                    # KADFP计算对齐误差并发送
                    start_time2 = time.time()
                    flow = estimator.estimate(curr_img, target_img)
                    drone_command = ""
                    drone_command= flow2drone(curr_img, target_img, flow, self.video_idx, intrinsic, curr_depth, 2000, KADFP_results_path)
                    cost2 = time.time() - start_time2
                    # print(f"flow estimate 執行時間：{cost2} 秒") 
                    print(f"total cost time: {cost1+cost2} 秒")
                    self.conn.sendall(struct.pack('!I', len(drone_command)))
                    self.conn.sendall(drone_command.encode('utf-8'))

                    # if kadfp_error<15:# 如果对齐成功，保存比较图片
                    #     compare_img_name = str(compare_save_path / f"compare_curr{self.curr_idx}_vid{self.video_idx}.png")
                    #     compare_img = np.concatenate((curr_img, target_img), axis=1)
                    #     cv2.imwrite(compare_img_name, compare_img)
                    #     print(f"保存比较图片到{compare_img_name}")

            except (ConnectionResetError, BrokenPipeError):
                # 客户端意外断开连接
                print(f"客户端 {self.addr} 意外断开连接")
                break
        self.conn.close()
        print(f"客户端 {self.addr} 已断开连接")
        
    def del_old_data(self):
        if os.path.exists(output):
            try:
                for folder_name in os.listdir(output):
                    if os.path.isdir(output / folder_name):
                        for old_img_name in os.listdir(output / folder_name):
                            os.unlink(output / folder_name / old_img_name)
                    else:
                        os.unlink(output / folder_name)
                print("删除旧数据成功")
            except Exception as e:
                print("删除旧数据失败")
                print(e)

    def send_depth(self, curr_depth, idx):
        # 准备要发送的数据
        image_data = cv2.imencode('.png', curr_depth)[1].tobytes()
        image_size = len(image_data)
        # 发送图像大小和idx
        self.conn.sendall(struct.pack('!II', image_size, idx))
        # 发送深度图数据
        self.conn.sendall(image_data)
        print('发送深度图，大小为', image_size)

    def recvall(self, n):
        # 辅助函数，接收指定大小的数据
        data = b''
        while len(data) < n:
            packet = self.conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data


if __name__ == '__main__':
    args = parse_args()
    # DepthAnythingV2 initialize
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{args.datasets}_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | [%(levelname)s] | %(message)s')
    # DepthAnythingV2 initialize end

    output = args.outputs
    output.mkdir(exist_ok = True, parents=True)
    KADFP_results_path = output / 'align_error/'
    KADFP_results_path.mkdir(exist_ok = True, parents=True)
    curr_frame_path = output / 'curr_frames/'
    curr_frame_path.mkdir(exist_ok = True, parents=True)
    disp_path = output / 'disp/'
    disp_path.mkdir(exist_ok = True, parents=True)
    # compare_save_path = output / 'compare/'
    # compare_save_path.mkdir(exist_ok = True, parents=True)

    # Camera parameters
    intrinsic = np.array([[517.417475, 0, 336.769936],
                          [0, 585.870258, 207.547167],
                          [0, 0, 1]])
    # KADFP initialize
    if args.kadfp:
        # list the standard configurations available
        print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
        print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

        # pick one of the configurations for extraction and matching
        # retrieval_conf = extract_features.confs['netvlad']
        feature_conf = extract_features.confs['superpoint_aachen']
        matcher_conf = match_features.confs['NN-superpoint']
        # intrinsic = dataset / 'queries' / 'intrinsic.txt'

        model_args = get_twins_args()
        logging.info('load RAFT model.')
        estimator = Flow_estimator(model_args)
        # KADFP initialize end
        
        target_video = args.target_video
        cap = cv2.VideoCapture(str(target_video))
        video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"影片長度：{video_len}")
        keyframe_idxs=[]

    server = Server('140.113.195.240', 8888)
    server.start()
    if args.kadfp:
        cap.release()
