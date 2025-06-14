from collections import defaultdict
from distutils.log import info
from pathlib import Path
from pprint import pformat
import argparse
import time
import cv2
from cv2 import log
import h5py
import torch
import torch.nn.functional as F
import logging
import socket
import select
import numpy as np 
from datetime import datetime
from skimage import io
import sys
import flow_vis
import configargparse
import argparse
import os
import random

from hloc import extractors
from hloc.utils.read_write_model import read_model, qvec2rotmat
from hloc.utils.io import list_h5_names
from hloc.utils.base_model import dynamic_load
from hloc.utils.flow_estimator import Flow_estimator
from hloc.utils.flow_match_commend import flow_to_drone

from ... import extract_features, match_features
from ... import pairs_from_covisibility, pairs_from_retrieval
from ... import colmap_from_nvm, triangulation, localize_sfm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=Path, default='datasets/test',
                    help='Path to the dataset, default: %(default)s')
parser.add_argument('--outputs', type=Path, default='outputs/test',
                    help='Path to the output directory, default: %(default)s')
parser.add_argument('--target', type=Path, default='target.jpg',
                    help='Path to the target image, default: %(default)s')
parser.add_argument('--num_loc', type=int, default=20,
                    help='Number of image pairs for loc, default: %(default)s')
args = parser.parse_args()

# Setup the paths
dataset = args.dataset
images = dataset / 'images_upright/'
images.mkdir(exist_ok = True, parents=True)

Stage_2_results_path = dataset / 'stage_2_res/'
Stage_2_results_path.mkdir(exist_ok = True, parents=True)

target_image_path = dataset / args.target
print(target_image_path)

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

stage2_count = 0

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
def load_netvlad():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(extractors, retrieval_conf['model']['name'])
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
    # print(result)
    result = np.array(['{:+.11f}'.format(result[n][0]) for n in range(len(result))], dtype='a11').reshape(-1, 1)
    # print(result)
    result = np.array2string(result)
    # print(result)

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

def get_twins_args():    
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument("--mixed_precision", type=str, default=True)
    parser.add_argument("--small", type=str, default=False)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--dim_corr", type=int, default=192)
    parser.add_argument("--dim_corr_coarse", type=int, default=64)
    parser.add_argument("--dim_corr_all", type=int, default=192)    
    parser.add_argument("--model", type=str, default="./hloc/pipelines/VBgps/pretrained_models/twins_one.pth")
    # parser.add_argument('--model', type=str, default="./snapshot/Twins_train_with_pretrain_smaple_len_250/checkpoint_latest.pth")  
    parser.add_argument("--fnet", type=str, default='twins')
    parser.add_argument("--twoscale", type=str, default=False)
    return parser.parse_known_args()[0]


model_args = get_twins_args()

logging.info('load RAFT model.')
estimator = Flow_estimator(model_args)
logging.info('load SfM model.')
sfm_model = read_model(str(reference_sfm))
logging.info('load db global feats.')
db_gdesc, db_names = load_db_gfeats(db_global_feats)
logging.info('load NetVlad model.')
netvlad = load_netvlad()
logging.info('load db local feats.')
db_local_name2ref = load_db_localfeats(db_feats)
query_info = parse_intrinsic(intrinsic)

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
    # Remove old data
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
            frame_idx = recv_info(sock, 10)
            if frame_idx is not None:
                frame_idx = int(frame_idx.decode())
                _ = recv_info(sock, 1)
                logging.info(f"frame index : {frame_idx}")
                Stage = recv_info(sock, 1).decode()
                # _ = recv_info(sock, 1)
                logging.info(f"Stage : {Stage}")
                print(Stage)
                print(type(Stage))
                img_size = recv_info(sock, 10).decode()
                _ = recv_info(sock, 1)
                logging.info(f"image size : {img_size}")
                query = recv_info(sock, int(img_size))
                query = np.fromstring(query, np.uint8)
                query_img = cv2.imdecode(query, cv2.IMREAD_COLOR)
                query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
                print(query_img.shape)
                current_image = query_img.copy()
                query_img = cv2.resize(query_img,(960,720))
                img_name = f"q_{frame_idx}.jpg"
                img_path = Path(images, img_name)
                cv2.imwrite(str(img_path), query_img )
                # cv2.imshow(query_img)
                if Stage == '1':
                    feat_output = Path(outputs, "query_data")
                    t_1 = time.monotonic()
                    features = extract_features.main(feature_conf, images, feat_output)
                    t_2 = time.monotonic()
                    global_descriptors = extract_features.main(retrieval_conf, images, feat_output, model=netvlad)
                    t_3 = time.monotonic()
                    
                    pairs_from_retrieval.main(
                        global_descriptors, loc_pairs, args.num_loc,
                        query_prefix='', db_prefix='', db_descriptors=db_gdesc, db_names=db_names) 
                    t_4 = time.monotonic()
                    
                    loc_matches = match_features.main(matcher_conf, loc_pairs, features, matches=
                                                    Path(outputs, f"{features.stem}_{matcher_conf['output']}_{loc_pairs.stem}"), 
                                                    features_ref=db_feats, name2ref = db_local_name2ref)
                    t_5 = time.monotonic()

                    poses = localize_sfm.main(
                        sfm_model,
                        #dataset / 'queries/*_time_queries_with_intrinsics.txt',
                        [(img_name, query_info)],
                        loc_pairs,
                        features,
                        loc_matches,
                        results,
                        covisibility_clustering=True)  # not required with SuperPoint+SuperGlue
                    #send pose
                    print(poses)
                    if img_name not in poses:
                        continue
                    qvec, tvec = poses[img_name]
                    result = pose_postproc(qvec, tvec)
                    print('===========================')
                    print(frame_idx)
                    print(result)
                    print(int(Stage))
                    print('===========================')
                    # sock.send(Stage.encode())
                    # sock.send(str('{:10}'.format(int(Stage))).encode())
                    sock.sendall(int(Stage).to_bytes(4, 'big'))
                    # print('result: ', result)
                    # print(len(result.encode()))
                    sock.send(result.encode())
                    sock.send(str('{:10}'.format(frame_idx)).encode())
                    t_6 = time.monotonic()

                    logs[c]["t_feats"].append(t_2 - t_1)   
                    logs[c]["t_global_feats"].append(t_3 - t_2)
                    logs[c]["t_retrieval"].append(t_4 - t_3)
                    logs[c]["t_match"].append(t_5 - t_4)
                    logs[c]["t_localize"].append(t_6 - t_5)
                    logs[c]["t_all"].append(t_6-t_1)

                    #remove img and feature data of query 
                    features.unlink()
                    global_descriptors.unlink()
                    img_path.unlink()


                elif Stage == '2':                        # Stage 2
                    current_image = cv2.resize(current_image,(640,480))  # rgb
        
                    current_image_bgr = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
                    # Save the image
                    # Define the output file path
                    output_path = os.path.join(Stage_2_results_path, "current" + str(stage2_count) + ".jpg")
                    cv2.imwrite(output_path, current_image_bgr)

                    target_image = io.imread(target_image_path)
                    start_time = time.time()
                    flow = estimator.estimate(current_image, target_image)
                    end_time  = time.time()
                    execution_time = end_time - start_time
                    print(f"flow estimate 執行時間：{execution_time} 秒") 
                    
                    Points_match, drone_commend = flow_to_drone(current_image, target_image, flow, Stage_2_results_path, stage2_count)
                    stage2_count += 1
                    #---------visualize flow ------------------------------------------
                    flow_np = flow.squeeze().detach().cpu().numpy()
                    flow_permuted = np.transpose(flow_np, (1, 2, 0))      
                    flow_visualized =  flow_vis.flow_to_color(flow_permuted, convert_to_bgr=True)
                    cv2.imwrite(os.path.join(Stage_2_results_path, "flow_visualized.png"), flow_visualized)
                    #------------------------------------------------------------------
                    # blend_cur_tar = cv2.addWeighted(current_image, 0.7, target_image, 0.7, 30)

                    # blend_cur_tar_bgr = cv2.cvtColor(blend_cur_tar, cv2.COLOR_RGB2BGR)
                    # # Save the image
                    # # Define the output file path
                    # output_path = os.path.join(Stage_2_results_path, "blend.jpg")

                    # # Save the image
                    # cv2.imwrite(output_path, blend_cur_tar_bgr)

                    #---------current(rgb) to brg and save-----------------------------
                    # Convert RGB to BGR
                    query_img_bgr = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
                    # # Save the image
                    # # Define the output file path
                    # output_path = os.path.join(Stage_2_results_path, "output.jpg")

                    # # Save the image
                    # cv2.imwrite(output_path, query_img_bgr)
                    #---------send stage number-----------------------------
                    
                    sock.sendall(int(Stage).to_bytes(4, 'big'))

                    #---------send a string-----------------------------
                    
                    # Just for test 
                    # random_int = random.randint(0, 5)
                    # drone_commend = test_commend[random_int]
                    # print(drone_commend)

                    # Send a string to the client
                    drone_commend_length = len(drone_commend)
                    print('drone_commend_length: ', drone_commend_length)
                    drone_commend_length_bytes = drone_commend_length.to_bytes(4, 'big')
                    sock.send(drone_commend_length_bytes)
                    sock.send(drone_commend.encode())

                    #---------send a image------------------------------
                    # Points_match_brg = cv2.cvtColor(Points_match, cv2.COLOR_RGB2BGR)
                    # # Encode the image as JPEG
                    # _, image_data = cv2.imencode('.jpg', Points_match_brg)
                    # image_size = len(image_data)
                    # print('image_size: ', image_size)

                    # # Send the image size to the client
                    # size_bytes = image_size.to_bytes(4, 'big')
                    # sock.send(size_bytes)

                    # # Send the image data to the client
                    # sock.sendall(image_data.tobytes())
                    #---------------------------------------------------
                #     sock.send(result.encode())
            else :
                logging.info(f"connection {c} ends")
                fds.remove(sock)
                sock.close()
                # with open('Hloc_time.txt', 'a') as f:
                #     now = datetime.now()
                #     flag = datetime.strftime(now, "%m-%d")
                #     f.write(flag+ ' ' + '\n')
                #     t_feats = logs[c]["t_feats"]
                #     t_global_feats = logs[c]["t_global_feats"]
                #     t_retrieval = logs[c]["t_retrieval"]
                #     t_match = logs[c]["t_match"]
                #     t_localize = logs[c]["t_localize"]
                #     t_all = logs[c]["t_all"]
                #     for log in zip(t_feats, t_global_feats, t_retrieval, t_match, t_localize):
                #         f.write(f"feat : {log[0]:.5f} global_feat: {log[1]:.5f} retrieval: {log[2]:.5f} match: {log[3]:.5f} localize: {log[4]:.5f}\n")
                #     f.write(f"\nAVG feat : {sum(t_feats)/len(t_feats):.5f} global_feat: {sum(t_global_feats)/len(t_global_feats):.5f} retrieval: {sum(t_retrieval)/len(t_retrieval):.5f} match: {sum(t_match)/len(t_match):.5f} localize: {sum(t_localize)/len(t_localize):.5f}\n")
                #     f.write(f"total AVG: {sum(t_all)/len(t_feats):.5f}\n")
        

    
