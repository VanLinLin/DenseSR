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
from losses import CharbonnierLoss, TVLoss
import os
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
import torch.nn.functional as F
from utils.loader import get_training_data, get_validation_data
import time
import argparse
import densesr_options
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

    def __call__(self, model, input_, point, normal, dino_net, device, dino_patch_size=16):
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
            h_size = H * dino_patch_size // 8

            w_size = W * dino_patch_size // 8
            
            UpSample_window = torch.nn.UpsamplingBilinear2d(size=(h_size, w_size))
            
            # Get DINO features
            with torch.no_grad():
                input_DINO = UpSample_window(input_pad)
                dino_features = dino_net.module.get_intermediate_layers(input_DINO, n=4, reshape=True)
            
            # Model inference
            with torch.amp.autocast(device_type='cuda'):
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
                # DINO_patch_size = 16 # Removed hardcode
                h_size = self.window_size * dino_patch_size // 8
                w_size = self.window_size * dino_patch_size // 8
                
                UpSample_window = torch.nn.UpsamplingBilinear2d(size=(h_size, w_size))
                
                # Get DINO features
                with torch.no_grad():
                    input_DINO = UpSample_window(input_window)
                    dino_features = dino_net.module.get_intermediate_layers(input_DINO, n=4, reshape=True)
                
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
        return output


######### parser ###########
opt = densesr_options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
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
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


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
# DINO Patch Size handling
if opt.dino_version == 'dinov3':
    DINO_patch_size = 16
    if opt.dino_model == 'vitl16':
        dino_model_name = 'dinov3_vitl16'
        dino_weights = 'dinov3/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        opt.dino_dim = 1024
    else:
        raise ValueError(f"For DINOv3, only 'vitl16' is supported. Got: {opt.dino_model}")
        
    DINO_Net = torch.hub.load('./dinov3', dino_model_name, source='local', weights=dino_weights)

elif opt.dino_version == 'dinov2':
    DINO_patch_size = 14
    # For DINOv2, we map 'vitl16' (user input/default) to 'dinov2_vitl14' or explicit 'vitl14'
    if 'vitl' in opt.dino_model:
        dino_model_name = 'dinov2_vitl14'
        opt.dino_dim = 1024
    else:
         raise ValueError(f"For DINOv2, only 'vitl' (vitl14) is implemented here. Got: {opt.dino_model}")
    
    # Load DINOv2 from local source
    # Assuming 'dinov2' directory is in current path and has hubconf.py or similar structure compatible with torch.hub.load
    # source='local' looks for hubconf.py in the directory.
    DINO_Net = torch.hub.load('./dinov2', dino_model_name, source='local', pretrained=False)
    
    # Check if we need to load weights manually or if pretrained=True handles it (normally pretrained=True downloads).
    # If the user has local weights, we might need to load them manually if not handled by hubconf with source='local'.
    # For now assuming we rely on hubconf or standard loading. 
    # BUT wait, the existing code for dinov3 used 'weights' arg which is specific to that repo's hubconf?
    # Let's check DINOv2 hubconf. It usually takes 'pretrained' arg.
    # If weights are local, we might need to load_state_dict. 
    # Let's assume for DINOv2 we load the model structure and the user might expect it to download or have it cached, 
    # OR we use the default 'pretrained=True' if we want weights.
    # However, source='local' often implies we want to run offline.
    # Let's set pretrained=True so it tries to load weights (from cache or download).
    DINO_Net = torch.hub.load('./dinov2', dino_model_name, source='local', pretrained=True)

else:
    raise ValueError(f"Unknown dino version: {opt.dino_version}")

model_restoration = utils.get_arch(opt)
model_restoration.to(device)


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
        lr = utils.load_optim(optimizer, path_chk_rest)
        utils.load_checkpoint(model_restoration,path_chk_rest)

    # new_lr = lr
    # if opt.optimizer.lower() == 'adam':
    #     optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
    # elif opt.optimizer.lower() == 'adamw':
    #     optimizer = optim.AdamW(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
    # else:
    #     raise Exception("Error optimizer...")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-5)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                  mode='min',
    #                                                  factor=0.85,
    #                                                  patience=10,
    #                                                  min_lr=1e-7)
    logging.info("------------------------------------------------------------------------------")
    logging.info(f"==> Resuming Training with learning rate:{utils.load_optim(optimizer, path_chk_rest)}")
    logging.info("------------------------------------------------------------------------------")
    # print(optimizer.param_groups[0]['lr'])

######### DDP ###########
model_restoration = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_restoration).to(device)
model_restoration = DDP(model_restoration, device_ids=[local_rank], output_device=local_rank)

DINO_Net.to(device)
DINO_Net.eval()
DINO_Net = DDP(DINO_Net, device_ids=[local_rank], output_device=local_rank)


# ######### Scheduler ###########
if not opt.resume:
    if opt.warmup:
        logging.info("Using warmup and cosine strategy!")
        warmup_epochs = opt.warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-5)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step()
# if not opt.resume:
#     if opt.warmup:
#         logging.info("Using warmup and cosine strategy!")
#         warmup_epochs = opt.warmup_epochs
#         scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-5)
#         scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
#         scheduler.step()
    else:
        step = 50
        logging.info("Using StepLR,step={}!".format(step))
        scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
        scheduler.step()

######### Loss ###########
criterion_restore = CharbonnierLoss().to(device)
criterion_tv = TVLoss(tv_loss_weight=0.1).to(device)
# criterion_perceptual = PerceptualLoss().to(device)

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
best_ssim = 0
best_epoch = 0
best_iter = 0
logging.info("\nEvaluation after every {} Iterations !!!\n".format(opt.eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()

index = 0
index = 0
# DINO_patch_size is already needed above for UpSample
# But notice UpSample is defined here in the main script loop scope.
# DINO_patch_size was defined here as 16.
# We set DINO_patch_size when loading the model.
# So we just use the variable we defined earlier.
img_multiple_of = 8 * opt.win_size

# the train_ps must be the multiple of win_size
UpSample = nn.UpsamplingBilinear2d(
    size=((int)(opt.train_ps * DINO_patch_size / 8), 
        (int)(opt.train_ps * DINO_patch_size / 8))
    )

Charbonnier_weight = 0.9
SSIM_weight = 0.002

for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0

    epoch_direct_loss = 0
    epoch_indirect_loss = 0
    train_id = 1
    epoch_ssim_loss = 0
    epoch_tv_loss = 0
    epoch_Charbonnier_loss = 0
    epoch_perceptual_loss = 0


    train_loader.sampler.set_epoch(epoch)

    train_bar = tqdm(train_loader, disable=dist.get_rank() != 0)

    for i, data in enumerate(train_bar, 0): 
        # zero_grad
        index += 1
        optimizer.zero_grad()
        target = data[0].to(device)
        input_ = data[1].to(device)
        point = data[2].to(device)
        normal = data[3].to(device)

        with torch.amp.autocast('cuda'):
            dino_mat_features = None
            with torch.no_grad():
                input_DINO = UpSample(input_)
                dino_mat_features = DINO_Net.module.get_intermediate_layers(input_DINO, n=4, reshape=True)

            restored = model_restoration(input_, dino_mat_features, point, normal)
            loss_restore = criterion_restore(restored, target)
            loss = 0.9 * loss_restore

            # print(f"loss_restore: {loss_restore}")
            # print(f"loss: {loss}")
            # print(f"loss_restore * 0.9: {loss_restore*0.9}")

            # tv_loss = criterion_tv(restored)

            # ssim_loss = 1 - ssim_metric(restored, target)

            #TODO
            # perceptual_value = criterion_perceptual(restored, target)


            # loss = 0.8 * loss_restore + 0.1 * tv_loss + 0.1 * ssim_loss  # 權重需要調整

            #TODO
            # loss = 0.7 * loss_restore + 0.1 * tv_loss + 0.1 * ssim_loss + 0.1 * perceptual_value

            # loss = Charbonnier_weight * loss_restore + SSIM_weight * ssim_loss



        loss_scaler(loss, optimizer,parameters=model_restoration.parameters())

        loss_list = utils.distributed_concat(loss, dist.get_world_size())
        # ssim_loss_list = utils.distributed_concat(ssim_loss, dist.get_world_size())
        # tv_loss_list = utils.distributed_concat(tv_loss, dist.get_world_size())
        Charbonnier_loss_list = utils.distributed_concat(loss_restore, dist.get_world_size())


        #TODO
        # perceptual_loss_list = utils.distributed_concat(perceptual_value, dist.get_world_size())
        loss = 0
        for ele in loss_list:
            loss += ele.item()


        # ssim_loss = sum(ele.item() for ele in ssim_loss_list) / dist.get_world_size()
        # tv_loss = sum(ele.item() for ele in tv_loss_list) / dist.get_world_size()
        Charbonnier_loss = sum(ele.item() for ele in Charbonnier_loss_list) / dist.get_world_size()

        #TODO
        # perceptual_loss = sum(ele.item() for ele in perceptual_loss_list) / dist.get_world_size()



        epoch_loss += loss
        # epoch_ssim_loss += (SSIM_weight * ssim_loss) / len(train_loader)
        # epoch_tv_loss += tv_loss / len(train_loader)
        epoch_Charbonnier_loss += (Charbonnier_weight * Charbonnier_loss) / len(train_loader)

        #TODO
        # epoch_perceptual_loss += perceptual_loss / len(train_loader)


        if dist.get_rank() == 0:
            train_bar.set_description(f"Train Epoch: [{epoch}/{opt.nepoch}] Loss: {loss:.4f}")
            tb_logger.add_scalar("train/loss", epoch_loss, epoch+1)
            # tb_logger.add_scalar("train/SSIM_Loss", epoch_ssim_loss, epoch+1)
            # tb_logger.add_scalar("TV_loss", epoch_tv_loss, epoch+1)
            tb_logger.add_scalar("train/Charbonnier_loss", epoch_Charbonnier_loss, epoch+1)

            #TODO
            # tb_logger.add_scalar("perceptual_loss", epoch_perceptual_loss, epoch+1)
            tb_logger.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch+1)
    train_bar.close()
    ################# Evaluation ########################
#     if (epoch + 1) % opt.eval_now == 0:
#         eval_shadow_rmse = 0
#         eval_nonshadow_rmse = 0
#         eval_rmse = 0
#         with torch.no_grad():
#             model_restoration.eval()
#             psnr_val_rgb = []

#             val_bar = tqdm(val_loader, disable=dist.get_rank() != 0)
#             val_bar.set_description(f'Validation Epoch {epoch}')

#             for _, data_val in enumerate(val_bar, 0):
#                 target = data_val[0].to(device)
#                 input_ = data_val[1].to(device)
#                 point = data_val[2].to(device)
#                 normal = data_val[3].to(device)
#                 filenames = data[4] 
#                 # Pad the input if not_multiple_of win_size * 8
#                 height, width = input_.shape[2], input_.shape[3]
#                 H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
#                     (width + img_multiple_of) // img_multiple_of) * img_multiple_of

#                 padh = H - height if height % img_multiple_of != 0 else 0
#                 padw = W - width if width % img_multiple_of != 0 else 0
#                 input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
#                 point = F.pad(point, (0, padw, 0, padh), 'reflect')
#                 normal = F.pad(normal, (0, padw, 0, padh), 'reflect')
#                 UpSample_val = nn.UpsamplingBilinear2d(
#                     size=((int)(input_.shape[2] * (DINO_patch_size / 8)), 
#                         (int)( input_.shape[3] * (DINO_patch_size / 8))))


#                 with torch.cuda.amp.autocast():

#                     # DINO_V2
#                     input_DINO = UpSample_val(input_)
#                     dino_mat_features = DINO_Net.module.get_intermediate_layers(input_DINO, n=4, reshape=True)
#                     restored = model_restoration(input_, dino_mat_features, point, normal)


#                 restored = torch.clamp(restored, 0.0, 1.0)
#                 restored = restored[:, : ,:height, :width]
#                 psnr_val_rgb.append(utils.batch_PSNR(restored, target, True))


#             psnr_val_rgb = sum(psnr_val_rgb) / len(val_loader)
#             psnr_val_rgb_list = utils.distributed_concat(psnr_val_rgb, dist.get_world_size())
#             psnr_val_rgb = 0
#             for ele in psnr_val_rgb_list:
#                 psnr_val_rgb += ele.item()

#             psnr_val_rgb = psnr_val_rgb / len(psnr_val_rgb_list)

#             if dist.get_rank() == 0:
#                 tb_logger.add_scalar("psnr", psnr_val_rgb, epoch)
#             val_bar.set_postfix(psnr=f'{sum(psnr_val_rgb)/len(psnr_val_rgb):.4f}')
#             val_bar.close()

#             for ele in restored:
#                 rgb_restored = ele.cpu().numpy().squeeze().transpose((1, 2, 0))
#                 utils.save_img(rgb_restored * 255.0, os.path.join(result_dir, filenames[0]))

#             if psnr_val_rgb > best_psnr:
#                 best_psnr = psnr_val_rgb
#                 best_epoch = epoch
#                 best_iter = i
#                 if dist.get_rank() == 0:
#                     torch.save({'epoch': epoch,
#                                 'state_dict': model_restoration.module.state_dict(),
#                                 'optimizer' : optimizer.state_dict()
#                                 }, os.path.join(model_dir,"model_best.pth"))

#             logging.info("[Ep %d it %d\t PSNR SIDD: %.4f\t ] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
#                     % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
#             logging.info("Now time is : {}".format(datetime.datetime.now().isoformat()))
#             model_restoration.train()
#             torch.cuda.empty_cache()
#     scheduler.step()
    
#     logging.info("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
#     if dist.get_rank() == 0:
#         torch.save({'epoch': epoch, 
#                     'state_dict': model_restoration.module.state_dict(),
#                     'optimizer' : optimizer.state_dict()
#                     }, os.path.join(model_dir,"model_latest.pth"))   

#         if epoch%opt.checkpoint == 0:
#             torch.save({'epoch': epoch, 
#                         'state_dict': model_restoration.module.state_dict(),
#                         'optimizer' : optimizer.state_dict()
#                         }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
# logging.info("Now time is : {}".format(datetime.datetime.now().isoformat()))


# 以下是sliding window
    if (epoch + 1) % opt.eval_now == 0:
        eval_shadow_rmse = 0
        eval_nonshadow_rmse = 0
        eval_rmse = 0
        with torch.no_grad():
            model_restoration.eval()
            psnr_val_rgb = []
            ssim_val_rgb = []
            val_loss_rgb = []
            val_charbonnier_loss_rgb = []
            val_ssim_loss_rgb = []

            val_bar = tqdm(val_loader, disable=dist.get_rank()!=0) # 只在主進程顯示
            val_bar.set_description(f'Validation Epoch {epoch}')

            for _, data_val in enumerate(val_bar, 0):
                target = data_val[0].to(device)
                input_ = data_val[1].to(device)
                point = data_val[2].to(device)
                normal = data_val[3].to(device)
                filenames = data_val[4]

                # 初始化滑動窗口推理
                sliding_window = SlidingWindowInference(
                    window_size=256,  # 與訓練相同的 patch size
                    overlap=64,       # 相應調整 overlap
                    img_multiple_of=8 * opt.win_size
                )

                # 使用滑動窗口進行推理
                with torch.amp.autocast(device_type='cuda'):
                    restored = sliding_window(
                        model=model_restoration,
                        input_=input_,
                        point=point,
                        normal=normal,
                        dino_net=DINO_Net,
                        device=device,
                        dino_patch_size=DINO_patch_size
                    )

                # 確保輸出在合理範圍內
                restored = torch.clamp(restored, 0.0, 1.0)

                # **計算 Loss**
                charbonnier_loss = criterion_restore(restored, target)
                ssim_loss = 1 - ssim_metric(restored, target)
                total_loss = Charbonnier_weight * charbonnier_loss + SSIM_weight * ssim_loss

                val_loss_rgb.append(total_loss.item())
                val_charbonnier_loss_rgb.append(charbonnier_loss.item())
                val_ssim_loss_rgb.append(ssim_loss.item())

                # **計算 PSNR 和 SSIM**
                psnr_val_rgb.append(utils.batch_PSNR(restored, target, True))
                ssim_val_rgb.append(ssim_loss.item())

                # 保存結果（只在主進程中保存）
                if dist.get_rank() == 0:
                    for idx, filename in enumerate(filenames):
                        rgb_restored = restored[idx].cpu().numpy().squeeze().transpose((1, 2, 0))
                        utils.save_img(rgb_restored * 255.0, os.path.join(result_dir, filename))

                val_bar.set_postfix(psnr=f'{sum(psnr_val_rgb)/len(psnr_val_rgb):.4f}', 
                                    ssim=f'{sum(ssim_val_rgb)/len(ssim_val_rgb):.4f}', 
                                    loss=f'{sum(val_loss_rgb)/len(val_loss_rgb):.4f}')

            # **計算平均 loss**
            psnr_val_rgb = sum(psnr_val_rgb) / len(val_loader)
            ssim_val_rgb = sum(ssim_val_rgb) / len(val_loader)
            val_loss_rgb = sum(val_loss_rgb) / len(val_loader)
            val_charbonnier_loss_rgb = sum(val_charbonnier_loss_rgb) / len(val_loader)
            val_ssim_loss_rgb = sum(val_ssim_loss_rgb) / len(val_loader)

            # **確保 loss 是 Tensor**
            psnr_val_rgb = torch.tensor(psnr_val_rgb, dtype=torch.float32, device=device)
            ssim_val_rgb = torch.tensor(ssim_val_rgb, dtype=torch.float32, device=device)
            val_loss_rgb = torch.tensor(val_loss_rgb, dtype=torch.float32, device=device)
            val_charbonnier_loss_rgb = torch.tensor(val_charbonnier_loss_rgb, dtype=torch.float32, device=device)
            val_ssim_loss_rgb = torch.tensor(val_ssim_loss_rgb, dtype=torch.float32, device=device)

            # **同步分布式 loss**
            val_loss_rgb_list = utils.distributed_concat(val_loss_rgb, dist.get_world_size())
            val_charbonnier_loss_rgb_list = utils.distributed_concat(val_charbonnier_loss_rgb, dist.get_world_size())
            val_ssim_loss_rgb_list = utils.distributed_concat(val_ssim_loss_rgb, dist.get_world_size())

            val_loss_rgb = sum(ele.item() for ele in val_loss_rgb_list) / len(val_loss_rgb_list)
            val_charbonnier_loss_rgb = sum(ele.item() for ele in val_charbonnier_loss_rgb_list) / len(val_charbonnier_loss_rgb_list)
            val_ssim_loss_rgb = sum(ele.item() for ele in val_ssim_loss_rgb_list) / len(val_ssim_loss_rgb_list)

            # **同步分布式 PSNR 和 SSIM**
            psnr_val_rgb_list = utils.distributed_concat(psnr_val_rgb, dist.get_world_size())
            ssim_val_rgb_list = utils.distributed_concat(ssim_val_rgb, dist.get_world_size())

            psnr_val_rgb = sum(ele.item() for ele in psnr_val_rgb_list) / len(psnr_val_rgb_list)
            ssim_val_rgb = sum(ele.item() for ele in ssim_val_rgb_list) / len(ssim_val_rgb_list)

            # **記錄到 TensorBoard**
            if dist.get_rank() == 0:
                tb_logger.add_scalar("val/psnr", psnr_val_rgb, epoch)
                tb_logger.add_scalar("val/ssim", ssim_val_rgb, epoch)
                tb_logger.add_scalar("val/loss", val_loss_rgb, epoch)
                tb_logger.add_scalar("val/Charbonnier_loss", val_charbonnier_loss_rgb, epoch)
                tb_logger.add_scalar("val/SSIM_loss", val_ssim_loss_rgb, epoch)

                # **保存最佳模型**
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model_restoration.module.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_best.pth"))

            # **輸出日誌**
            logging.info("[Ep %d it %d\t PSNR: %.4f\t] ----  [Best_Epoch %d Best_Iter %d Best_PSNR %.4f] " \
                    % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
            logging.info("[Ep %d it %d\t SSIM: %.4f\t] ----  [Best_Epoch %d Best_Iter %d Best_SSIM %.4f] " \
                    % (epoch, i, ssim_val_rgb, best_epoch, best_iter, best_ssim))
            logging.info("[Ep %d it %d\t Validation Loss: %.4f\t]" % (epoch, i, val_loss_rgb))
            logging.info("[Ep %d it %d\t Charbonnier Loss: %.4f\t]" % (epoch, i, val_charbonnier_loss_rgb))
            logging.info("[Ep %d it %d\t SSIM Loss: %.4f\t]" % (epoch, i, val_ssim_loss_rgb))
            logging.info("Now time is : {}".format(datetime.datetime.now().isoformat()))

            # **恢復訓練模式並清理顯存**
            model_restoration.train()
            torch.cuda.empty_cache()
    # scheduler.step(val_loss_rgb)  # with ReduceLROnPlateau
    scheduler.step()

    
    # logging.info("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    logging.info("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(
    epoch, time.time()-epoch_start_time, epoch_loss, optimizer.param_groups[0]['lr']))
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








