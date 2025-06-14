from moviepy.editor import ImageSequenceClip
import argparse
import cv2
import numpy as np
import os
import torch
import logging
import socket
from pathlib import Path
from collections import defaultdict
from distutils.log import info
import struct

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

sys.path.append('../hloc')
from utils.read_write_model import read_model, qvec2rotmat

from utils.flow_estimator import Flow_estimator
from utils.flow_match_commend import flow_to_drone,flow_to_error

import extract_features, match_features
import traceback
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Metric3D Metric Depth Estimation')
    
    parser.add_argument('--kadfp', action='store_true', help='activate KADFP model')
    parser.add_argument('--outputs', type=Path, default='../outputs/', 
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--target_video', type=Path, default='../datasets/testvideo/test2.mp4',
                        help='Path to the target image, default: %(default)s')
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

def get_metric3d_depth(metric3d_model, img_path, rgb, size, pad_info, intrinsic):
    print(f'Progress {img_path}')
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
        depth[depth > 60000] = 0
        depth = depth.astype(np.uint16)

        output_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(disp_path, "{}_disp.png".format(output_name))
        print(f'Save to {output_path}')
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
        self.del_old_data()# 删除旧数据
        
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

                # Metric3D 输入前处理
                start_time1 = time.time()
                rgb = cv2.resize(curr_img, (int(w_origin * scale), int(h_origin * scale)), interpolation=cv2.INTER_LINEAR)
                # remember to scale intrinsic, hold depth
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
                curr_depth = get_metric3d_depth(metric3d_model, img_path, rgb,  (h_origin,w_origin), pad_info, intrinsic_scaled)
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
                    cap.set(cv2.CAP_PROP_POS_FRAMES,self.video_idx)
                    ret, img = cap.read()
                    target_img = cv2.resize(img,(640,480))
                    
                    # KADFP计算对齐误差并发送
                    start_time2 = time.time()
                    flow = estimator.estimate(curr_img, target_img)
                    kadfp_dir, kadfp_error = flow_to_error(curr_img, target_img, flow, self.video_idx,self.curr_idx, KADFP_results_path)
                    drone_command = kadfp_dir+" "+str(f'{kadfp_error:.2f}')
                    cost2 = time.time() - start_time2
                    # print(f"flow estimate 執行時間：{cost2} 秒") 
                    print(f"total cost time: {cost1+cost2} 秒")
                    self.conn.sendall(struct.pack('!I', len(drone_command)))
                    self.conn.sendall(drone_command.encode('utf-8'))

                    if kadfp_error<15:# 如果对齐成功，保存比较图片
                        compare_img_name = str(compare_save_path / f"compare_curr{self.curr_idx}_vid{self.video_idx}.png")
                        compare_img = np.concatenate((curr_img, target_img), axis=1)
                        cv2.imwrite(compare_img_name, compare_img)
                        print(f"保存比较图片到{compare_img_name}")

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
    output = args.outputs
    output.mkdir(exist_ok = True, parents=True)

    KADFP_results_path = output / 'align_error/'
    KADFP_results_path.mkdir(exist_ok = True, parents=True)
    video_frame_path = output / 'video_frames/'
    video_frame_path.mkdir(exist_ok = True, parents=True)
    curr_frame_path = output / 'curr_frames/'
    curr_frame_path.mkdir(exist_ok = True, parents=True)
    disp_path = output / 'disp/'
    disp_path.mkdir(exist_ok = True, parents=True)
    compare_save_path = output / 'compare/'
    compare_save_path.mkdir(exist_ok = True, parents=True)

    # Camera parameters
    # intrinsic = [517.417475, 585.870258, 336.769936, 207.547167]
    intrinsic = [535.4, 539.2, 320.1, 247.6]
    h_origin, w_origin = 480, 640
    
    # metric3d initialize
    metric3d_model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    metric3d_model.cuda().eval()
    input_size = (616, 1064) # for vit model
    scale = min(input_size[0] / h_origin, input_size[1] / w_origin)
    padding = [123.675, 116.28, 103.53]
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    # KADFP initialize
    if args.kadfp:
        # list the standard configurations available
        print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
        print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')
        feature_conf = extract_features.confs['superpoint_aachen']
        matcher_conf = match_features.confs['NN-superpoint']
        model_args = get_twins_args()
        logging.info('load RAFT model.')
        estimator = Flow_estimator(model_args)
        # KADFP initialize end
        
        target_video = args.target_video
        cap = cv2.VideoCapture(str(target_video))
        video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"影片長度：{video_len}")

    server = Server('140.113.195.240', 8888)
    server.start()
    if args.kadfp:
        cap.release()
