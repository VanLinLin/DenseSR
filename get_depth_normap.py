import sys
sys.path.append('Depth-Anything-V2')

import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from depth_anything_v2.dpt import DepthAnythingV2
from pathlib import Path
from tqdm.auto import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Generate depth and normal maps from images')
    parser.add_argument('--source_root', type=str, default='test_dir', 
                        help='Root directory containing the images')
    parser.add_argument('--model_path', type=str, 
                        default='depth_anything_v2_vitl.pth',
                        help='Path to the depth model checkpoint')
    return parser.parse_args()


def generate_depth_maps(source_root, model_path):
    source_root = Path(source_root)
    origin = source_root / 'origin'
    to_thermal_list = [origin]

    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]).cuda()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    thermal_path = source_root / 'depth'

    with torch.inference_mode():
        for to_thermal_item in to_thermal_list:
            folder_name = to_thermal_item.stem
            dst_path = thermal_path

            dst_path.mkdir(parents=True, exist_ok=True)
            
            bar = tqdm(to_thermal_item.glob('*'))

            for image_path in bar:
                try:
                    raw_img = cv2.imread(str(image_path))
                    depth = model.infer_image(raw_img)  # HxW raw depth map

                    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                    depth = depth.astype(np.uint8)

                    print(depth.shape)
                    np.save(f'{dst_path}/{image_path.stem}.npy', depth)

                except Exception as e:
                    print(e)
                    continue
    
    return thermal_path


def calculate_normal_map(img_path: Path, ksize=5):
    # 讀取深度圖
    depth = np.load(img_path).astype(np.float32)

    # 計算 X、Y 方向的梯度
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=ksize)

    # 假設 Z 軸方向為 -1
    dz = np.ones_like(dx) * -1

    # 組合成法向量 (Nx, Ny, Nz)
    normals = np.stack((dx, dy, dz), axis=-1)

    # 進行歸一化
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= (norm + 1e-6)  # 避免除零錯誤

    # 轉換為 0-255 的 RGB 影像 (HWC)
    normal_map = (normals + 1) / 2 * 255
    normal_map = normal_map.astype("uint8")

    normal_map = normal_map.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

    return normal_map


def generate_normal_maps(source_root, ksize=5):
    source_root = Path(source_root)
    depth_root = source_root / 'depth'
    normal_root = source_root / 'normal'
    normal_root.mkdir(parents=True, exist_ok=True)

    bar = tqdm(list(depth_root.glob('*.npy')))

    for depth_img_path in bar:
        img_name = depth_img_path.name

        normal_map = calculate_normal_map(depth_img_path, ksize=ksize)

        np.save(f'{normal_root}/{img_name}', normal_map)


def main():
    args = parse_args()
    
    print(f"Generating depth maps from images in {args.source_root}")
    depth_path = generate_depth_maps(args.source_root, args.model_path)
    
    print(f"Generating normal maps from depth maps")
    generate_normal_maps(args.source_root)
    
    print("Processing complete!")


if __name__ == "__main__":
    main()