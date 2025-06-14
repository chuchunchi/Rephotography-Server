import os
import torch
import cv2
import numpy as np
from pathlib import Path
import time

def get_metric3d_depth(metric3d_model, img_name, rgb, output_path, intrinsic):
    print(f'Progress {img_name}')
    with torch.no_grad():
        pred_depth,_,_= metric3d_model.inference({'input': rgb})
        print(f'1pred_depth: {pred_depth.shape}')
        pred_depth = pred_depth.squeeze()
        print(f'2pred_depth: {pred_depth.shape}')
        pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
        print(f'3pred_depth: {pred_depth.shape}')
        pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
        print(f'4pred_depth: {pred_depth.shape}')
        output_path = os.path.join(output_path, "{}_disp.png".format(str(img_name).split('.')[0]))
        canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
        pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
        pred_depth = torch.clamp(pred_depth, 0, 300)
        depth = pred_depth.cpu().numpy()
        
        print(depth.max(),depth.min())
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65535
        # print(depth)
        depth = depth.astype(np.uint16)
        print(f'Save to {output_path}')
        cv2.imwrite(output_path, depth)
        return depth
    
if __name__ == '__main__':
    # Load the model
    metric3d_model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
    metric3d_model.cuda().eval()
    img_path = Path('../outputs/1207_1/curr_frames/')
    output_path = Path('../outputs/metric3d/disp/')
    output_path.mkdir(exist_ok = True, parents=True)
    for img_name in os.listdir(img_path):
        rgb_origin = cv2.imread(os.path.join(img_path, img_name))[:, :, ::-1]
        start_time1 = time.time()
        #### ajust input size to fit pretrained model
        # keep ratio resize
        # intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
        intrinsic = [517.417475, 585.870258, 336.769936, 207.547167]
        input_size = (616, 1064) # for vit model
        # input_size = (544, 1216) # for convnext model
        padding = [123.675, 116.28, 103.53]
        h, w = rgb_origin.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        # remember to scale intrinsic, hold depth
        intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
        
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        
        
        #### normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)
        rgb = rgb[None, :, :, :].cuda()
        print(rgb.shape)
        get_metric3d_depth(metric3d_model, img_name, rgb, output_path, intrinsic)
        cost1 = time.time() - start_time1
        print(f"執行時間：{cost1} 秒") 