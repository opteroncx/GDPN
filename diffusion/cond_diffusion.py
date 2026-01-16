import argparse
import sys
from unittest.mock import patch
sys.path.append('ilvr_adm')
import torch
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import load_data
from torchvision import utils
from resizer import Resizer, linear
import math
import time
import numpy as np
import torch.nn as nn

def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        ll = self.LL(x)
        lh = self.LH(x)
        hl = self.HL(x)
        hh = self.HH(x)
        return ll, lh, hl, hh

class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH):
        # if self.option_unpool == 'sum':
        #     return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        return self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH)

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0
        self.num_freq = len(mapper_x)
        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter

class Learnable_LPF(torch.nn.Module):
    def __init__(self, resizers, dct_h=56, dct_w=56, reduction = 16, freq_sel_method = 'top16'):
        super(Learnable_LPF, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
        channel = 64
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.in_block = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1,1),
        )
        self.out_block = nn.Sequential(
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,3,3,1,1),            
        )
        self.resizers = resizers
        self.haar = Haar_LPF()
        self.alpha = torch.tensor(0.1)

    def forward(self, x):
        # let each channel of the convolution layer learn a different filter
        # residual arch: up(down(x)) + net(x)
        down, up = self.resizers
        resize_lpf = up(down(x))
        haar_lpf = self.haar(x)
        x = self.in_block(x)
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        x = x * y.expand_as(x)
        x = self.out_block(x)
        return self.alpha*x+(1-self.alpha)*haar_lpf
        # return x
        # return x + haar_lpf

class Haar_LPF(nn.Module):
    def __init__(self):
        super(Haar_LPF, self).__init__()
        self.wave = WavePool(3)
        self.unwave = WaveUnpool(3)
        self.eps = 1e-6

    def forward(self, x):
        ll, lh, hl, hh = self.wave(x)
        ll, lh, hl, hh = self.unwave(ll, lh, hl, hh)
        return ll

class Configs():
    def __init__(self):
        self.clip_denoised=True
        self.num_samples=1
        self.batch_size=1
        self.down_N=4
        self.ange_t=20
        self.use_ddim=False
        self.base_samples='/home/user/Documents/datasets/CelebA-HQ/GT'
        self.model_path='models/ffhq_10m.pt'
        self.save_dir='output/celeb-64x64'
        self.save_latents=False
        self.image_size=256
        self.num_channels=128
        self.num_res_blocks=1
        self.num_heads=4
        self.num_heads_upsample=-1
        self.num_head_channels=64
        self.attention_resolutions='16'
        self.channel_mult=''
        self.dropout=0.0
        self.class_cond=False
        self.use_checkpoint=False
        self.use_scale_shift_norm=True
        self.resblock_updown=True
        self.use_fp16=False
        self.use_new_attention_order=False
        self.learn_sigma=True
        self.diffusion_steps=1000
        self.noise_schedule='linear'
        self.timestep_respacing='100'
        self.use_kl=False
        self.predict_xstart=False
        self.rescale_timesteps=False
        self.rescale_learned_sigmas=False

class Diffusion(object):
    def __init__(self,scale=16,num=1):
        print("creating diffusion model...")
        self.device = 'cuda'
        args = Configs()
        scale_n_dict = {
            16: 4,
            32: 8,
            64: 16
        }
        args.down_N = scale_n_dict[scale]
        args.batch_size = num
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(torch.load('/home/user/Documents/codes/FaceGAN/ilvr_adm/models/ffhq_10m.pt'))
        model.to(self.device)
        model.eval()
        model.requires_grad_(False)

        shape = (args.batch_size, 3, args.image_size, args.image_size)
        shape_d = (args.batch_size, 3, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
        worKing_mode = 'Haar_LPF'
        resize_kernel = 'linear'
        if worKing_mode == 'ILVR':
            down = Resizer(shape, 1 / args.down_N,kernel=resize_kernel).to(next(model.parameters()).device)
            up = Resizer(shape_d, args.down_N,kernel=resize_kernel).to(next(model.parameters()).device)
            self.resizers = {'mode':'ILVR','content':(down, up)}
        elif worKing_mode == 'Haar_LPF':
            lpf = Haar_LPF()
            lpf = lpf.cuda()
            self.resizers = {'mode':'Haar_LPF','content':lpf}
        elif worKing_mode == 'Learnable_LPF':
            down = Resizer(shape, 1 / args.down_N,kernel=resize_kernel).to(next(model.parameters()).device)
            up = Resizer(shape_d, args.down_N,kernel=resize_kernel).to(next(model.parameters()).device)
            resizers = (down, up)
            lpf = Learnable_LPF(resizers)
            lpf = lpf.cuda()
            lpf.requires_grad_(True)
            self.resizers = {'mode':'Learnable_LPF','content':lpf}            
        self.model = model
        self.diffusion = diffusion
        self.args = args

    def forward(self, x):
        model_kwargs = {"ref_img":x}
        model_kwargs = {k: v.to(self.device) for k, v in model_kwargs.items()}
        if self.args.use_ddim:
            sample = self.diffusion.ddim_sample_loop(
                self.model,
                (self.args.batch_size, 3, 256, 256),
                clip_denoised=True,
                model_kwargs=model_kwargs,
                resizers=self.resizers,
                range_t=20
            )
        else:
            sample = self.diffusion.p_sample_loop(
                self.model,
                (self.args.batch_size, 3, 256, 256),
                clip_denoised=True,
                model_kwargs=model_kwargs,
                resizers=self.resizers,
                range_t=20
            )
        return sample

if __name__ == '__main__':
    scale = 64
    num=1
    diffusion = Diffusion(scale,num)
    path = './531.jpg'
    from skimage import io,transform
    img = io.imread(path)
    img = transform.resize(img,(16,16),order=3)
    io.imsave('LR.png',transform.resize(img,(256,256),order=0))
    img = transform.resize(img,(256,256),order=0)
    img = img * 255
    img = img /127.5 -1
    img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).float()
    img = torch.cat([img]*num,dim=0)
    print(img.shape)
    st = time.time()
    res = diffusion.forward(img)
    et = time.time()-st
    print(res.shape,'time used',et)
    for i in range(num):
        resi = res[i]
        resi = resi.cpu().permute(1,2,0).numpy()
        resi = (resi+1)*127.5
        io.imsave('diffusion_example_%d.png'%i,resi)