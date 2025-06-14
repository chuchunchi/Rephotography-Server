import threading

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
import struct

sys.path.append('../hloc')
import extractors
from utils.read_write_model import qvec2rotmat

from utils.flow_estimator import Flow_estimator
from utils.flow_match_commend import flow_to_drone

import extract_features, match_features
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--video_only', action='store_true', help='only estimate the pose of the video frames')
    parser.add_argument('--input-size', type=int, default=518)
    # parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    parser.add_argument('--outputs', type=Path, default='../outputs/xr/', #1206_drone',
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
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65535.0

    output_path = os.path.join(disp_path, "{}_disp.png".format(output_name))
    cv2.imwrite(output_path, depth.astype(np.uint16))
    return output_path

def pose_postproc(qvec ,tvec):
    R_t = np.eye(4)
    R = qvec2rotmat(qvec)
    R_t[:3, :3] = R.T
    R_t[:3, 3] = -R.T @ tvec
    result = np.array(R_t).reshape(-1,1)
    result = np.array(['{:+.11f}'.format(result[n][0]) for n in range(len(result))], dtype='a11').reshape(-1, 1)
    result = np.array2string(result)
    return result


class Server:
    def __init__(self, host='', port=12345):
        self.host = host
        self.port = port
        self.sock = None
        self.conn = None
        self.addr = None
        self.string_lock = threading.Lock()
        self.unsent_string = None  # 存储未发送的字符串
        self.curr_idx = 0
        self.video_idx = 0
        self.target_img_cp = None
        self.compare_counter = 0
        self.this_frame_is_ok = True
        self.video_over = False

    def start(self):
        while True:
            try:
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
                self.conn.close()
                break
            except Exception as e:
                # 捕获其他未知异常，确保服务器不会崩溃
                traceback.print_exc()
                self.conn.close()
                break

    def handle_client(self):
        loop_count = 0
        threadId = 0
        while not self.video_over:
            loop_count += 1
            try:
                # 删除旧数据
                # self.del_old_data()

                # 接收图像大小
                data = self.recvall(4)
                if not data:
                    break
                image_size = struct.unpack('!I', data)[0]
                # 接收图像数据
                image_data = self.recvall(image_size)
                if not image_data:
                    break
                print('收到图像，大小为', image_size)
                nparr = np.frombuffer(image_data, np.uint8)
                curr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                curr_img = cv2.resize(curr_img,(640,480))
                img_name = f"q{self.curr_idx}.png"
                self.curr_idx += 1
                img_path = str(curr_frame_path / img_name)
                cv2.imwrite(img_path, curr_img)
                # 执行DepthanythingV2得到深度图
                depth_path = get_depth(args, img_path, curr_img)
                
                # KADFP计算指令字符串
                drone_command = b''
                string_included = False
                if not args.video_only:
                    print(f'开启KADFP线程{threadId}')
                    threading.Thread(target=self.kadfp_command, args=(curr_img, compare_save_path, threadId)).start()
                    threadId += 1
                    # 检查上一个字符串是否已发送
                    with self.string_lock:
                        if self.unsent_string is not None:
                            string_included = True
                            drone_command = self.unsent_string.encode('utf-8')
                            self.unsent_string = None  # 标记字符串已发送
                        else:
                            string_included = False
                            drone_command = b''

                # 发送深度图和指令字符串
                self.send_response(depth_path, drone_command, string_included)

            except (ConnectionResetError, BrokenPipeError):
                # 客户端意外断开连接
                print(f"客户端 {self.addr} 意外断开连接")
                break
        self.conn.close()

    def del_old_data(self):
        for old_data in os.listdir(Path(curr_frame_path)):
            os.unlink(Path(curr_frame_path) / old_data)
        if not args.video_only:
            for old_img_name in os.listdir(images):
                os.unlink(images / old_img_name)
            # for old_item in os.listdir(outputs):
            #     if '.txt' in old_item:
            #         if old_item == 'pairs-exhaustive.txt':
            #             pass
            #         else:
            #             os.unlink(outputs / old_item)
            #     elif '.7_pairs-query-netvlad20' in old_item:
            #         os.unlink(outputs / old_item)

    def kadfp_command(self,curr_img,compare_save_path,threadId):
        # target_img = cv2.imread("./assets/images/0052.jpg", cv2.IMREAD_COLOR)
        if self.target_img_cp is not None:
            target_img = self.target_img_cp.copy()
        if self.this_frame_is_ok or self.compare_counter==5:
            print("读取下一帧")
            self.this_frame_is_ok = False
            self.compare_counter = 0
            ret, img = cap.read()
            ret, img = cap.read()
            if not ret:
                self.video_over = True
                cap.release()
                print('视频结束')
            x,y,w,h=80,0,480,480
            target_img = img[y:y+h, x:x+w]
            target_img = cv2.resize(target_img,(640,480))
            self.target_img_cp = target_img.copy()
            self.video_idx+=2
            
        self.compare_counter += 1        
        drone_command=""
        start_time = time.time()
        flow = estimator.estimate(curr_img, target_img)
        end_time  = time.time()
        execution_time = end_time - start_time
        print(f"flow estimate 執行時間：{execution_time} 秒") 
        
        _, drone_command = flow_to_drone(curr_img, target_img, flow, KADFP_results_path, self.video_idx)
        print(drone_command)
        dir = drone_command.split()[0]
        if dir=='original':
            compare_img = np.concatenate((curr_img, target_img), axis=1)
            cv2.imwrite(str(compare_save_path / f"compare_{threadId}.png"), compare_img)
            print(f'this frame({self.video_idx}) is ok, start to campare next frame({self.video_idx+2}) ')
            self.this_frame_is_ok = True
            drone_command = ""
        #---------visualize flow ------------------------------------------
        # flow_np = flow.squeeze().detach().cpu().numpy()
        # flow_permuted = np.transpose(flow_np, (1, 2, 0))
        # flow_visualized =  flow_vis.flow_to_color(flow_permuted, convert_to_bgr=True)
        # cv2.imwrite(os.path.join(Stage_2_results_path, "flow_visualized.png"), flow_visualized)

        with self.string_lock:
            if self.unsent_string is None:
                self.unsent_string = drone_command

        print(f"线程{threadId}结束")


    def send_response(self, depth_path, drone_command, string_included):
        # 准备要发送的数据
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        image_data = cv2.imencode('.png', depth_image)[1].tobytes()
        image_size = len(image_data)
        string_size = len(drone_command)
        # 打包头部信息
        header = struct.pack('!II?', image_size, string_size, string_included)
        # 发送头部信息
        self.conn.sendall(header)
        # 发送深度图数据
        self.conn.sendall(image_data)
        # 如果包含字符串，发送字符串数据
        if string_included:
            self.conn.sendall(drone_command)
        print('发送深度图，大小为', image_size)
        if string_included:
            print('发送指令，大小为', string_size)

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
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | [%(levelname)s] | %(message)s')

    # KADFP initialize
    if not args.video_only:
        output = args.outputs
        images = output / 'images_upright/'
        images.mkdir(exist_ok = True, parents=True)

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

        target_video = args.target_video

        print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
        print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

        feature_conf = extract_features.confs['superpoint_aachen']
        matcher_conf = match_features.confs['NN-superpoint']
        # intrinsic = dataset / 'queries' / 'intrinsic.txt'

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

        cap = cv2.VideoCapture(str(target_video))
        frame_count = 0
    # KADFP initialize end


    server = Server('140.113.195.240', 9999)
    server.start()
