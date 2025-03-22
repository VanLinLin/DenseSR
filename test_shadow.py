import numpy as np
import os
import argparse
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import random
# from utils.loader import get_validation_data
from utils.loader import get_test_data
import utils
import cv2
import torch.distributed as dist
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='test_dir',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./output_dir',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='ACVLab_shadow.pth'
                    ,type=str, help='Path to weights')
# parser.add_argument('--arch', default='ShadowFormer', type=str, help='arch')
parser.add_argument('--arch', type=str, default='ShadowFormerFreq', help='archtechture')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', default=False, help='Save denoised images in result directory')
parser.add_argument('--cal_metrics', action='store_true', default=False, help='Measure denoised images with GT')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=16, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')

parser.add_argument('--train_ps', type=int, default=256, help='patch size of training sample')
parser.add_argument("--local-rank", type=int)

args = parser.parse_args()

local_rank = args.local_rank
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
device = torch.device("cuda", local_rank)


class SlidingWindowInference:
    def __init__(self, window_size=512, overlap=64, img_multiple_of=64):
        self.window_size = window_size
        self.overlap = overlap
        self.img_multiple_of = img_multiple_of
        
    def _pad_input(self, x, h_pad, w_pad):
        """Handle padding using reflection padding"""
        return F.pad(x, (0, w_pad, 0, h_pad), 'reflect')

    def __call__(self, model, input_, point, normal, dino_net, device):
        # Save original dimensions
        original_height, original_width = input_.shape[2], input_.shape[3]
        # print(f"Original size: {original_height}x{original_width}")
        
        # Calculate minimum dimensions needed (at least window_size and multiple of img_multiple_of)
        H = max(self.window_size, 
               ((original_height + self.img_multiple_of - 1) // self.img_multiple_of) * self.img_multiple_of)
        W = max(self.window_size, 
               ((original_width + self.img_multiple_of - 1) // self.img_multiple_of) * self.img_multiple_of)
        # print(f"Target padded size: {H}x{W}")
        
        # Calculate required padding
        padh = H - original_height
        padw = W - original_width
        # print(f"Padding: h={padh}, w={padw}")
        
        # Pad all inputs
        input_pad = self._pad_input(input_, padh, padw)
        point_pad = self._pad_input(point, padh, padw)
        normal_pad = self._pad_input(normal, padh, padw)
        
        # If image was smaller than window_size, process it as a single window
        if original_height <= self.window_size and original_width <= self.window_size:
            # print("Image smaller than window size, processing as single padded window")
            
            # For DINO features
            DINO_patch_size = 14
            h_size = H * DINO_patch_size // 8
            w_size = W * DINO_patch_size // 8
            
            UpSample_window = torch.nn.UpsamplingBilinear2d(size=(h_size, w_size))
            
            # Get DINO features
            with torch.no_grad():
                input_DINO = UpSample_window(input_pad)
                dino_features = dino_net.module.get_intermediate_layers(input_DINO, 4, True)
            
            # Model inference
            with torch.cuda.amp.autocast():
                restored = model(input_pad, dino_features, point_pad, normal_pad)
            
            # Crop back to original size
            output = restored[:, :, :original_height, :original_width]
            return output
        
        # For larger images, proceed with sliding window approach
        stride = self.window_size - self.overlap
        h_steps = (H - self.window_size + stride - 1) // stride + 1
        w_steps = (W - self.window_size + stride - 1) // stride + 1
        # print(f"Steps: h={h_steps}, w={w_steps}")
        
        # Create output tensor and counter
        output = torch.zeros_like(input_pad)
        count = torch.zeros_like(input_pad)
        
        for h_idx in range(h_steps):
            for w_idx in range(w_steps):
                # Calculate current window position
                h_start = min(h_idx * stride, H - self.window_size)
                w_start = min(w_idx * stride, W - self.window_size)
                h_end = h_start + self.window_size
                w_end = w_start + self.window_size
                
                # Get current window
                input_window = input_pad[:, :, h_start:h_end, w_start:w_end]
                point_window = point_pad[:, :, h_start:h_end, w_start:w_end]
                normal_window = normal_pad[:, :, h_start:h_end, w_start:w_end]
                
                # print(f"Processing window at ({h_idx}, {w_idx}): {input_window.shape}")
                
                # For DINO features
                DINO_patch_size = 14
                h_size = self.window_size * DINO_patch_size // 8
                w_size = self.window_size * DINO_patch_size // 8
                
                UpSample_window = torch.nn.UpsamplingBilinear2d(size=(h_size, w_size))
                
                # Get DINO features
                with torch.no_grad():
                    input_DINO = UpSample_window(input_window)
                    dino_features = dino_net.module.get_intermediate_layers(input_DINO, 4, True)
                
                # Model inference
                with torch.cuda.amp.autocast():
                    restored = model(input_window, dino_features, point_window, normal_window)
                
                # Create weight mask for smooth transition
                weight = torch.ones_like(restored)
                if self.overlap > 0:
                    # Create gradual weights for overlap regions
                    for i in range(self.overlap):
                        ratio = i / self.overlap
                        weight[:, :, i, :] *= ratio
                        weight[:, :, -(i+1), :] *= ratio
                        weight[:, :, :, i] *= ratio
                        weight[:, :, :, -(i+1)] *= ratio
                
                # Accumulate results and weights
                output[:, :, h_start:h_end, w_start:w_end] += restored * weight
                count[:, :, h_start:h_end, w_start:w_end] += weight
        
        # Normalize output
        output = output / (count + 1e-6)
        
        # Crop back to original size
        output = output[:, :, :original_height, :original_width]
        return output


utils.mkdir(args.result_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)

g = torch.Generator()
g.manual_seed(1234)

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
######### Model ###########
model_restoration = utils.get_arch(args)
model_restoration.to(device)
model_restoration.eval()
DINO_Net = torch.hub.load('./dinov2', 'dinov2_vitl14', source='local')
DINO_Net.to(device)
DINO_Net.eval()
######### Load ###########
utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

######### DDP ###########

model_restoration = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_restoration).to(device)
model_restoration = DDP(model_restoration, device_ids=[local_rank], output_device=local_rank)
DINO_Net = DDP(DINO_Net, device_ids=[local_rank], output_device=local_rank)

######### Test ###########
img_multiple_of = 8 * args.win_size
DINO_patch_size = 14

def UpSample(img):
    upsample = nn.UpsamplingBilinear2d(
        size=((int)(img.shape[2] * (DINO_patch_size / 8)), 
            (int)(img.shape[3] * (DINO_patch_size / 8))))
    return upsample(img)

img_options_train = {'patch_size':args.train_ps}
test_dataset = get_test_data(args.input_dir, False)
test_sampler = DistributedSampler(test_dataset, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=0, sampler=test_sampler, drop_last=False, worker_init_fn=worker_init_fn, generator=g)
with torch.no_grad():
    psnr_val_rgb_list = []
    psnr_val_mask_list = []
    ssim_val_rgb_list = []
    rmse_val_rgb_list = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
            # rgb_gt = data_test[0].numpy().squeeze().transpose((1, 2, 0))
            rgb_noisy = data_test[1].to(device)
            point = data_test[2].to(device)
            normal = data_test[3].to(device)
            filenames = data_test[4]

            # Pad the input if not_multiple_of win_size * 8
            # height, width = rgb_noisy.shape[2], rgb_noisy.shape[3]
            # H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
            #     (width + img_multiple_of) // img_multiple_of) * img_multiple_of

            # padh = H - height if height % img_multiple_of != 0 else 0
            # padw = W - width if width % img_multiple_of != 0 else 0
            # rgb_noisy = F.pad(rgb_noisy, (0, padw, 0, padh), 'reflect')
            # point = F.pad(point, (0, padw, 0, padh), 'reflect')
            # normal = F.pad(normal, (0, padw, 0, padh), 'reflect')
            # print(f'{rgb_noisy.shape=} {point.shape=} {normal.shape=}')
            # UpSample_val = nn.UpsamplingBilinear2d(
            #     size=((int)(rgb_noisy.shape[2] * (DINO_patch_size / 8)), 
            #         (int)(rgb_noisy.shape[3] * (DINO_patch_size / 8))))
            # with torch.cuda.amp.autocast():
            #     # DINO_V2
            #     input_DINO = UpSample_val(rgb_noisy)
            #     dino_mat_features = DINO_Net.module.get_intermediate_layers(input_DINO, 4, True)
            #     rgb_restored = model_restoration(rgb_noisy, dino_mat_features, point, normal)
            sliding_window = SlidingWindowInference(
                window_size=512,  # 與訓練相同的 patch size
                overlap=64,       # 相應調整 overlap
                img_multiple_of=8 * args.win_size
            )

            with torch.cuda.amp.autocast():
                rgb_restored = sliding_window(
                    model=model_restoration,
                    input_=rgb_noisy,
                    point=point,
                    normal=normal,
                    dino_net=DINO_Net,
                    device=device
                )

        
            rgb_restored = torch.clamp(rgb_restored, 0.0, 1.0)
            # rgb_restored = rgb_restored[:, : ,:height, :width]
            rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))
            

            if args.save_images:
                utils.save_img(rgb_restored * 255.0, os.path.join(args.result_dir, filenames[0]))


