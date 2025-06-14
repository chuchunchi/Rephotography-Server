import os
from moviepy.editor import ImageSequenceClip

def save2video(img_path,save_name,sort_basis,fps=30):
    if len(os.listdir(img_path)) > 0:
        # 獲取文件夾內所有圖片文件名，並按名稱中的序號排序
        imgs = sorted([f for f in os.listdir(img_path) if f.endswith('.png')], key=lambda x: int(x.split('.')[0][sort_basis:]))
    
        # 獲取完整的圖片文件路徑列表
        imgs = [os.path.join(img_path, img) for img in imgs]

        # 創建視頻剪輯
        clip = ImageSequenceClip(imgs, fps)

        # 將視頻剪輯寫入文件
        clip.write_videofile(save_name, codec='libx264')

    else:
        print(f"No images in {img_path} !")

if __name__ == '__main__':
    img_path = '/home/covis-lab-00/xr/data/outputs/0113_4/curr_frames'
    save_name = '/home/covis-lab-00/xr/data/outputs/0113_4/result.mp4'
    sort_basis = 1
    save2video(img_path,save_name,sort_basis,fps=30)