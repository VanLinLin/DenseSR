import os
import utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import datetime
from losses import CharbonnierLoss
import os
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
import torch.nn.functional as F
from utils.loader import get_training_data, get_validation_data
import time
import argparse
import densesr_options_plateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm

from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure



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
        
        # 检查图像是否小于窗口大小，如果是，直接处理整个图像
        if original_height < self.window_size or original_width < self.window_size:
            # print(f"Image smaller than window size, processing entire image directly")
            # 确保尺寸是img_multiple_of的倍数
            H = ((original_height + self.img_multiple_of - 1) // self.img_multiple_of) * self.img_multiple_of
            W = ((original_width + self.img_multiple_of - 1) // self.img_multiple_of) * self.img_multiple_of
            
            # 计算需要的填充
            padh = H - original_height if original_height % self.img_multiple_of != 0 else 0
            padw = W - original_width if original_width % self.img_multiple_of != 0 else 0
            
            # 填充输入
            input_pad = self._pad_input(input_, padh, padw)
            point_pad = self._pad_input(point, padh, padw)
            normal_pad = self._pad_input(normal, padh, padw)
            
            # 对于DINO特征
            DINO_patch_size = 14
            h_size = H * DINO_patch_size // 8
            w_size = W * DINO_patch_size // 8
            
            UpSample_full = torch.nn.UpsamplingBilinear2d(size=(h_size, w_size))
            
            # 获取DINO特征
            with torch.no_grad():
                input_DINO = UpSample_full(input_pad)
                dino_features = dino_net.module.get_intermediate_layers(input_DINO, 4, True)
            
            # 模型推理
            with torch.amp.autocast('cuda'):
                restored = model(input_pad, dino_features, point_pad, normal_pad)
            
            # 裁剪回原始尺寸
            restored = restored[:, :, :original_height, :original_width]
            
            return restored
        
        # 原来的代码，处理大于窗口大小的图像
        # Ensure size is multiple of img_multiple_of
        H = ((original_height + self.img_multiple_of - 1) // self.img_multiple_of) * self.img_multiple_of
        W = ((original_width + self.img_multiple_of - 1) // self.img_multiple_of) * self.img_multiple_of
        # print(f"Padded size: {H}x{W}")
        
        # Calculate required padding
        padh = H - original_height if original_height % self.img_multiple_of != 0 else 0
        padw = W - original_width if original_width % self.img_multiple_of != 0 else 0
        # print(f"Padding: h={padh}, w={padw}")
        
        # Pad all inputs
        input_pad = self._pad_input(input_, padh, padw)
        point_pad = self._pad_input(point, padh, padw)
        normal_pad = self._pad_input(normal, padh, padw)
        
        # Calculate stride and steps
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
                
                # print(f"Window at ({h_idx}, {w_idx}): {input_window.shape}")
                
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
                with torch.amp.autocast('cuda'):
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
        
        # print(f"Final output size: {output.shape}")
        return output

# def validate_sliding_window(val_loader, model, criterion, device, DINO_Net):
#     model.eval()
#     sliding_window = SlidingWindowInference(window_size=256, overlap=32)
    
#     psnr_val_rgb = []
#     for _, data_val in enumerate(val_loader, 0):
#         target = data_val[0].to(device)
#         input_ = data_val[1].to(device)
#         point = data_val[2].to(device)
#         normal = data_val[3].to(device)
#         filenames = data_val[4]
        
#         with torch.no_grad():
#             restored = sliding_window(model, input_, point, normal, DINO_Net, device)
#             restored = torch.clamp(restored, 0.0, 1.0)
            
#             # 計算 PSNR
#             psnr_val_rgb.append(utils.batch_PSNR(restored, target, True))
            
#             # 保存結果
#             for idx, filename in enumerate(filenames):
#                 rgb_restored = restored[idx].cpu().numpy().squeeze().transpose((1, 2, 0))
#                 utils.save_img(rgb_restored * 255.0, os.path.join(result_dir, filename))
    
#     return psnr_val_rgb



######### parser ###########
opt = densesr_options_plateau.Options().init(argparse.ArgumentParser(description='NTIRE shadow remove')).parse_args()
print(opt)
local_rank = opt.local_rank
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='gloo')
device = torch.device("cuda", local_rank)
if opt.debug == True:
    opt.eval_now = 2

######### Logs dir ###########
dir_name = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(dir_name, opt.save_dir, opt.arch+opt.env+datetime.datetime.now().isoformat(timespec='minutes'))
logname = os.path.join(log_dir, datetime.datetime.now().isoformat(timespec='minutes')+'.txt') 

result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
tensorlog_dir  = os.path.join(log_dir, 'tensorlog')
if dist.get_rank() == 0:
    utils.mkdir(log_dir)
    utils.mkdir(result_dir)
    utils.mkdir(model_dir)
    utils.mkdir(tensorlog_dir)
    utils.mknod(logname)
    tb_logger = SummaryWriter(log_dir=tensorlog_dir)

####### just allow one process to print info to log
if dist.get_rank() == 0:
    logging.basicConfig(filename=logname,level=logging.INFO if dist.get_rank() in [-1, 0] else logging.WARN)
    torch.distributed.barrier()
else:
    torch.distributed.barrier()
    logging.basicConfig(filename=logname,level=logging.INFO if dist.get_rank() in [-1, 0] else logging.WARN)

logging.info(opt)
logging.info(f"Now time is : {datetime.datetime.now().isoformat()}")
########### Set Seeds ###########
random.seed(1234 + dist.get_rank())
np.random.seed(1234 + dist.get_rank())
torch.manual_seed(1234 + dist.get_rank())
torch.cuda.manual_seed(1234 + dist.get_rank())
torch.cuda.manual_seed_all(1234 + dist.get_rank())

def worker_init_fn(worker_id):
    random.seed(1234 + worker_id + dist.get_rank())

g = torch.Generator()
g.manual_seed(1234 + dist.get_rank())

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True


######### Model ###########
model_restoration = utils.get_arch(opt)
model_restoration.to(device)
DINO_Net = torch.hub.load('./dinov2', 'dinov2_vitl14', source='local')
logging.info(str(model_restoration) + '\n')


######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")

######### Resume ###########
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    if dist.get_rank() == 0:
        print("載入checkpoint")
        utils.load_checkpoint(model_restoration, path_chk_rest)
        
        # 載入optimizer但不使用其中的學習率
        optimizer_state = torch.load(path_chk_rest, map_location=device)['optimizer']
        optimizer.load_state_dict(optimizer_state)
        
        # 設置自定義學習率
        custom_lr = opt.lr_initial_l2 # 你可以在這裡直接設置想要的學習率
        for param_group in optimizer.param_groups:
            param_group['lr'] = custom_lr
            
        print(f'設置自定義學習率: {custom_lr}')

    # 改為ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # 因為我們是基於PSNR優化，所以使用max模式
        factor=0.85,  # 學習率減半
        patience=10,  # 3個epoch無改善後調整學習率
        verbose=True
    )
    logging.info("------------------------------------------------------------------------------")
    logging.info(f"==> Resuming Training with custom learning rate: {optimizer.param_groups[0]['lr']}")
    logging.info("==> Using ReduceLROnPlateau scheduler")
    logging.info("------------------------------------------------------------------------------")
    print(optimizer.param_groups[0]['lr'])
    

######### DDP ###########
model_restoration = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_restoration).to(device)
# model_restoration = DDP(model_restoration, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
model_restoration = DDP(model_restoration, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

DINO_Net.to(device)
DINO_Net.eval()
# DINO_Net = DDP(DINO_Net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
DINO_Net = DDP(DINO_Net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)


# ######### Scheduler ###########
if not opt.resume:
    if opt.warmup:
        logging.info("Using warmup and cosine strategy!")
        warmup_epochs = opt.warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-5)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step()
    else:
        step = 50
        logging.info("Using StepLR,step={}!".format(step))
        scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
        scheduler.step()

######### Loss ###########
criterion_Charbonnier = CharbonnierLoss().to(device)
# 添加L2 Loss
criterion_L2 = nn.MSELoss().to(device)
# criterion_Perceptual = PerceptualLoss().to(device)

######### DataLoader ###########
logging.info('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train, opt.debug)
train_sampler = DistributedSampler(train_dataset, shuffle=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size // dist.get_world_size(), 
        num_workers=opt.train_workers, sampler=train_sampler, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn,
        generator=g )

val_dataset = get_validation_data(opt.val_dir, opt.debug)
val_sampler = DistributedSampler(val_dataset, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size= 3 // dist.get_world_size(),
        num_workers=opt.eval_workers, sampler=val_sampler, pin_memory=False, drop_last=False, worker_init_fn=worker_init_fn,
        generator=g)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
logging.info(f"Sizeof training set: {len_trainset} sizeof validation set: {len_valset}")

######### train ###########
logging.info("===> Start Epoch {} End Epoch {}".format(start_epoch,opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
logging.info("\nEvaluation after every {} Iterations !!!\n".format(opt.eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()

index = 0
DINO_patch_size = 14
img_multiple_of = 8 * opt.win_size

# the train_ps must be the multiple of win_size
UpSample = nn.UpsamplingBilinear2d(
    size=((int)(opt.train_ps * DINO_patch_size / 8), 
        (int)(opt.train_ps * DINO_patch_size / 8))
    )


for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0

    epoch_perceptual_loss = 0
    epoch_charbonnier_loss = 0

    epoch_direct_loss = 0
    epoch_indirect_loss = 0
    train_id = 1
    epoch_ssim_loss = 0

    train_loader.sampler.set_epoch(epoch)

    train_bar = tqdm(train_loader, disable=dist.get_rank()!=0) # 只在主進程顯示
    train_bar.set_description(f'Training Epoch {epoch}')

    for i, data in enumerate(train_bar, 0): 
        # zero_grad
        index += 1
        optimizer.zero_grad()
        target = data[0].to(device)
        input_ = data[1].to(device)
        point = data[2].to(device)
        normal = data[3].to(device)

        # print(f'{target.shape=}')
        # print(f'{input_.shape=}')
        # print(f'{point.shape=}')
        # print(f'{normal.shape=}')

        # print(f'{target.unique()=}')
        # print(f'{input_.unique()=}')
        # print(f'{point.unique()=}')
        # print(f'{normal.unique()=}')

        with torch.amp.autocast('cuda'):
            dino_mat_features = None
            with torch.no_grad():
                input_DINO = UpSample(input_)
                dino_mat_features = DINO_Net.module.get_intermediate_layers(input_DINO, 4, True)

            restored = model_restoration(input_, dino_mat_features, point, normal)

        
            # print(f'{restored.shape=}')
            # print(f'{restored.unique()=}')

            charbonnier_loss = criterion_Charbonnier(restored, target)
            
            l2_loss = criterion_L2(restored, target)
            
            # 計算感知損失
            # perceptual_loss = criterion_Perceptual(restored, target)
            

            # print(f'{loss_restore.shape=}')
            # print(f'{loss_restore.unique()=}')
            # print(f'{loss_restore=}')



            # loss = 0.9 * charbonnier_loss + 0.1 * perceptual_loss  # 這裡的權重可以調整
            # loss = 0.9 * charbonnier_loss  # 這裡的權重可以調整
            loss = 0.7 * charbonnier_loss + 0.3 * l2_loss  # 這裡的權重可以調整
            # print(f'{loss=}')


        # 執行優化器步驟
        loss_scaler(loss, optimizer, parameters=model_restoration.parameters())

        # 收集分布式環境中的損失值
        loss_value = loss.detach()  # 取得損失值，不計算梯度
        charbonnier_value = charbonnier_loss.detach()
        l2_value = l2_loss.detach()
        # perceptual_value = perceptual_loss.detach()
    
    
        # 使用您的distributed_concat函數收集所有進程的損失
        loss_list = utils.distributed_concat(loss_value, dist.get_world_size())
        charbonnier_list = utils.distributed_concat(charbonnier_value, dist.get_world_size())
        l2_list = utils.distributed_concat(l2_value, dist.get_world_size())
        

        # print('*'*10, f'{loss.shape=}')


        # 計算所有進程的損失總和
        loss_sum = 0
        charbonnier_sum = 0
        l2_sum = 0
        
        for loss_item in loss_list:
            loss_sum += loss_item.item()
        for char_item in charbonnier_list:
            charbonnier_sum += char_item.item()
        for l2_item in l2_list:
            l2_sum += l2_item.item()

        epoch_loss += loss_sum
        epoch_charbonnier_loss += charbonnier_sum
        epoch_l2_loss = l2_sum  # 新增L2 loss累積
        # epoch_perceptual_loss += perceptual_sum


        # 在主進程中記錄TensorBoard
        if dist.get_rank() == 0:  # 每100個batch記錄一次
            # global_step = (epoch - 1) * len(train_loader) + i
            tb_logger.add_scalar("train/loss", loss_sum, epoch+1)
            tb_logger.add_scalar("train/charbonnier_loss", charbonnier_sum, epoch+1)
            tb_logger.add_scalar("train/l2_loss", l2_sum, epoch+1)  # 新增L2 loss記錄
            tb_logger.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch+1)

        # 更新進度條
        train_bar.set_postfix(
            loss=f'{loss_sum:.4f}', 
            char=f'{charbonnier_sum:.4f}',
            l2=f'{l2_sum:.4f}',  # 新增L2損失顯示
            lr=f'{optimizer.param_groups[0]["lr"]:.6f}'
        )
    # # 每個epoch結束後記錄平均損失
    # if dist.get_rank() == 0:
    #     tb_logger.add_scalar("epoch/train_loss", epoch_loss / len(train_loader), epoch)
    #     tb_logger.add_scalar("epoch/train_charbonnier_loss", epoch_charbonnier_loss / len(train_loader), epoch)
    #     tb_logger.add_scalar("epoch/train_perceptual_loss", epoch_perceptual_loss / len(train_loader), epoch)
    ################# Evaluation ########################
    if (epoch + 1) % opt.eval_now == 0:
        eval_shadow_rmse = 0
        eval_nonshadow_rmse = 0
        eval_rmse = 0
        with torch.no_grad():
            model_restoration.eval()
            psnr_val_rgb = []
            ssim_val_rgb = []
            val_loss_list = []
            val_charbonnier_loss_list = []
            val_perceptual_loss_list = []
            val_l2_loss_list = []

            val_bar = tqdm(val_loader, disable=dist.get_rank()!=0) # 只在主進程顯示
            val_bar.set_description(f'Validation Epoch {epoch}')

            for _, data_val in enumerate(val_bar, 0):
                target = data_val[0].to(device)
                input_ = data_val[1].to(device)
                point = data_val[2].to(device)
                normal = data_val[3].to(device)
                filenames = data_val[4]

                print(f'{target.shape=}, {input_.shape=}, {point.shape=}, {normal.shape=}')

                # 初始化滑動窗口推理
                sliding_window = SlidingWindowInference(
                    window_size=512,  # 改為與訓練相同的patch size
                    overlap=64,       # 相應調整overlap
                    img_multiple_of=8 * opt.win_size
                )

                # 使用滑動窗口進行推理
                with torch.amp.autocast('cuda'):
                    # 準備DINO特徵提取的上採樣
                    DINO_patch_size = 14
                    height, width = input_.shape[2], input_.shape[3]
                    h_size = int(height * (DINO_patch_size / 8))
                    w_size = int(width * (DINO_patch_size / 8))
                    UpSample_val = nn.UpsamplingBilinear2d(size=(h_size, w_size))

                    # 使用滑動窗口進行推理
                    restored = sliding_window(
                        model=model_restoration,
                        input_=input_,
                        point=point,
                        normal=normal,
                        dino_net=DINO_Net,
                        device=device
                    )
                    
                    charbonnier_loss = criterion_Charbonnier(restored, target)
                    l2_loss = criterion_L2(restored, target)
                    # perceptual_loss = criterion_Perceptual(restored, target)
                    # print(f'{charbonnier_loss=}, {perceptual_loss=}')
                    # total_loss = 0.8 * charbonnier_loss + 0.2 * perceptual_loss
                    
                    total_loss = 0.7 * charbonnier_loss + 0.3 * l2_loss
                    
                    # 計算PSNR
                    current_psnr = utils.batch_PSNR(restored, target, True)
                    
                    # 計算SSIM（如果需要）
                    # ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
                    # current_ssim = ssim_module(restored, target)
                    
                    # print(f'{current_psnr=}, {current_ssim=}, {total_loss=}, {charbonnier_loss=}, {perceptual_loss=}')
                    # print(f'{current_psnr=}, {total_loss=}, {charbonnier_loss=}')

                # 記錄各項指標
                psnr_val_rgb.append(current_psnr)
                # ssim_val_rgb.append(current_ssim.item())  # 確保是標量
                val_loss_list.append(total_loss.item())
                val_charbonnier_loss_list.append(charbonnier_loss.item())
                val_l2_loss_list.append(l2_loss.item())
                # val_perceptual_loss_list.append(perceptual_loss.item())

                # 確保輸出在合理範圍內
                restored = torch.clamp(restored, 0.0, 1.0)

                # 計算PSNR
                psnr_val_rgb.append(utils.batch_PSNR(restored, target, True))

                # 保存結果（只在主進程中保存）
                if dist.get_rank() == 0:
                    for idx, filename in enumerate(filenames):
                        rgb_restored = restored[idx].cpu().numpy().squeeze().transpose((1, 2, 0))
                        utils.save_img(rgb_restored * 255.0, os.path.join(result_dir, filename))

                # 更新進度條
                current_psnr_avg = sum(psnr_val_rgb) / len(psnr_val_rgb)
                # current_ssim_avg = sum(ssim_val_rgb) / len(ssim_val_rgb)
                # val_bar.set_postfix(psnr=f'{current_psnr_avg:.4f}', ssim=f'{current_ssim_avg:.4f}')
                val_bar.set_postfix(psnr=f'{current_psnr_avg:.4f}')


            # 計算本進程的平均指標
            psnr_avg = sum(psnr_val_rgb) / len(psnr_val_rgb) if psnr_val_rgb else 0
            # ssim_avg = sum(ssim_val_rgb) / len(ssim_val_rgb) if ssim_val_rgb else 0
            loss_avg = sum(val_loss_list) / len(val_loss_list) if val_loss_list else 0
            charbonnier_avg = sum(val_charbonnier_loss_list) / len(val_charbonnier_loss_list) if val_charbonnier_loss_list else 0
            l2_avg = sum(val_l2_loss_list) / len(val_l2_loss_list) if val_l2_loss_list else 0
            # perceptual_avg = sum(val_perceptual_loss_list) / len(val_perceptual_loss_list) if val_perceptual_loss_list else 0
            # print(f'{psnr_avg=}, {ssim_avg=}, {loss_avg=}, {charbonnier_avg=}, {perceptual_avg=}')
            # print(f'{psnr_avg=}, {loss_avg=}, {charbonnier_avg=}')
        
        
            # 在分布式環境中同步評估指標 (將浮點數轉換為張量)
            psnr_tensor = torch.tensor([psnr_avg]).cuda()
            # ssim_tensor = torch.tensor([ssim_avg]).cuda()
            loss_tensor = torch.tensor([loss_avg]).cuda()
            charbonnier_tensor = torch.tensor([charbonnier_avg]).cuda()
            l2_tensor = torch.tensor([l2_avg]).cuda()
            # perceptual_tensor = torch.tensor([perceptual_avg]).cuda()
            # print(f'{psnr_tensor=}, {ssim_tensor=}, {loss_tensor=}, {charbonnier_tensor=}, {perceptual_tensor=}')
            # print(f'{psnr_tensor=}, {loss_tensor=}, {charbonnier_tensor=}')
            
            # 收集所有進程的指標
            psnr_list = utils.distributed_concat(psnr_tensor, dist.get_world_size())
            # ssim_list = utils.distributed_concat(ssim_tensor, dist.get_world_size())
            loss_list = utils.distributed_concat(loss_tensor, dist.get_world_size())
            charbonnier_list = utils.distributed_concat(charbonnier_tensor, dist.get_world_size())
            l2_list = utils.distributed_concat(l2_tensor, dist.get_world_size())
            # perceptual_list = utils.distributed_concat(perceptual_tensor, dist.get_world_size())
        
            
            # # 在分布式環境中同步評估指標
            # psnr_val_rgb_gather = [torch.zeros_like(torch.tensor(psnr_val_rgb)).cuda() for _ in range(dist.get_world_size())]
            # torch.distributed.all_gather(psnr_val_rgb_gather, torch.tensor(psnr_val_rgb).cuda())
            # psnr_val_rgb_all = []
            # for tensor in psnr_val_rgb_gather:
            #     psnr_val_rgb_all.extend(tensor.cpu().numpy())

            # 計算所有進程的平均指標
            psnr_val_rgb = sum([x.item() for x in psnr_list]) / len(psnr_list)
            # ssim_val_rgb = sum([x.item() for x in ssim_list]) / len(ssim_list)
            val_loss = sum([x.item() for x in loss_list]) / len(loss_list)
            val_charbonnier_loss = sum([x.item() for x in charbonnier_list]) / len(charbonnier_list)
            val_l2_loss = sum([x.item() for x in l2_list]) / len(l2_list)
            # val_perceptual_loss = sum([x.item() for x in perceptual_list]) / len(perceptual_list)


            # 在主進程中記錄TensorBoard和保存最佳模型
            if dist.get_rank() == 0:
                # 記錄所有指標到TensorBoard
                tb_logger.add_scalar("val/psnr", psnr_val_rgb, epoch)
                # tb_logger.add_scalar("val/ssim", ssim_val_rgb, epoch)
                tb_logger.add_scalar("val/loss", val_loss, epoch)
                tb_logger.add_scalar("val/charbonnier_loss", val_charbonnier_loss, epoch)
                tb_logger.add_scalar("val/l2_loss", val_l2_loss, epoch)
                # tb_logger.add_scalar("val/perceptual_loss", val_perceptual_loss, epoch)
                

                # 保存最佳模型
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model_restoration.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),  # 也保存調度器狀態
                        'best_psnr': best_psnr,
                        # 'best_ssim': ssim_val_rgb  # 記錄最佳SSIM
                    }, os.path.join(model_dir, "model_best.pth"))
                    
                    # 記錄最佳模型的所有指標
                    with open(os.path.join(model_dir, "best_metrics.txt"), 'w') as f:
                        f.write(f"Epoch: {best_epoch}\n")
                        f.write(f"PSNR: {best_psnr:.4f}\n")
                        # f.write(f"SSIM: {ssim_val_rgb:.4f}\n")
                        f.write(f"Loss: {val_loss:.4f}\n")
                        f.write(f"Time: {datetime.datetime.now().isoformat()}\n")

            # 輸出日誌
            logging.info("[Ep %d it %d\t PSNR: %.4f\t] ----  [best_Ep %d best_it %d Best_PSNR %.4f] " \
                    % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
            # logging.info("Val loss: %.4f, Char loss: %.4f, Perc loss: %.4f" \
            #         % (val_loss, val_charbonnier_loss, val_perceptual_loss))
            logging.info("Now time is : {}".format(datetime.datetime.now().isoformat()))
            
            # 恢復訓練模式並清理顯存
            model_restoration.train()
            torch.cuda.empty_cache()
    if opt.resume or not opt.warmup:  # 直接使用ReduceLROnPlateau的情況
        scheduler.step(psnr_val_rgb)
    else:  # 使用GradualWarmupScheduler時
        if epoch > opt.warmup_epochs:  # 在warmup結束後，正確調用ReduceLROnPlateau
            scheduler.step(psnr_val_rgb)
        else:
            scheduler.step()  # warmup階段正常step
    
    logging.info("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tL2 Loss: {:.4f}\tLearningRate {:.6f}".format(
        epoch, time.time()-epoch_start_time, epoch_loss, epoch_l2_loss, optimizer.param_groups[0]['lr']))
    if dist.get_rank() == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.module.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_latest.pth"))   

        if epoch%opt.checkpoint == 0:
            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.module.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
logging.info("Now time is : {}".format(datetime.datetime.now().isoformat()))





