# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from mmcv import Config
from mmedit.models.backbones.sr_backbones.rrdb_net import RRDB
from mmedit.models.builder import build_component
from mmedit.models.common import PixelShufflePack, make_layer
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from skimage import io
from diffusion import cond_diffusion

class FusionBlock(nn.Module):
    def __init__(self,num_channels):
        super().__init__()
        self.conv = nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True)
    def forward(self,x):
        out = self.conv(x)
        return out
'''
Codes are build based on GLEAN
'''
class GLEANStyleGANv2(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 img_channels=3,
                 rrdb_channels=64,
                 num_rrdbs=23,
                 style_channels=512,
                 num_mlps=8,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01,
                 default_style_mode='mix',
                 eval_style_mode='single',
                 mix_prob=0.9,
                 pretrained=None,
                 bgr2rgb=False):

        super().__init__()
        # latent bank (StyleGANv2), with weights being fixed
        print('loading StyleGANv2')
        self.generator = build_component(
            dict(
                type='StyleGANv2Generator',
                out_size=out_size,
                style_channels=style_channels,
                num_mlps=num_mlps,
                channel_multiplier=channel_multiplier,
                blur_kernel=blur_kernel,
                lr_mlp=lr_mlp,
                default_style_mode=default_style_mode,
                eval_style_mode=eval_style_mode,
                mix_prob=mix_prob,
                pretrained=pretrained,
                bgr2rgb=bgr2rgb))
        self.generator.requires_grad_(False)

        self.in_size = in_size
        self.style_channels = style_channels
        channels = self.generator.channels

        # encoder
        num_styles = int(np.log2(out_size)) * 2 - 2
        encoder_res = [2**i for i in range(int(np.log2(in_size)), 1, -1)]
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    img_channels, rrdb_channels, num_blocks=num_rrdbs),
                nn.Conv2d(
                    rrdb_channels, channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        for res in encoder_res:
            in_channels = channels[res]
            if res > 4:
                out_channels = channels[res // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, num_styles * style_channels))
            self.encoder.append(block)

        ################### ADD ##############
        # encoder_res = [64, 32, 16, 8, 4]
        encoder_res_diffusion = [256, 128, 64, 32, 16, 8, 4]
        diffusion_channels = {
            256:128,
            128:256,
            64:512,
            32:512,
            16:512,
            8:512,
            4:512
        }
        # print(encoder_res)
        self.encoder_diffusion = nn.ModuleList()
        self.encoder_diffusion.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    img_channels, rrdb_channels, num_blocks=3),
                nn.Conv2d(
                    rrdb_channels, diffusion_channels[256], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        for res in encoder_res_diffusion:
            in_channels = diffusion_channels[res]
            if res > 4:
                out_channels = diffusion_channels[res // 2]
                dblock = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                dblock = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, num_styles * style_channels))
            self.encoder_diffusion.append(dblock)
        ########################################################
        # additional modules for StyleGANv2
        self.fusion_out = nn.ModuleList()
        self.fusion_diffusion = nn.ModuleList()
        self.fusion_skip = nn.ModuleList()
        for res in encoder_res[::-1]:
            num_channels = channels[res]
            self.fusion_out.append(
                nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True))
            self.fusion_diffusion.append(
                FusionBlock(num_channels)
            )
            self.fusion_skip.append(
                nn.Conv2d(num_channels + 3, 3, 3, 1, 1, bias=True))

        # decoder
        decoder_res = [
            2**i
            for i in range(int(np.log2(in_size)), int(np.log2(out_size) + 1))
        ]
        self.decoder = nn.ModuleList()
        for res in decoder_res:
            if res == in_size:
                in_channels = channels[res]
            else:
                in_channels = 2 * channels[res]

            if res < out_size:
                out_channels = channels[res * 2]
                self.decoder.append(
                    PixelShufflePack(
                        in_channels, out_channels, 2, upsample_kernel=3))
            else:
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, 64, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(64, img_channels, 3, 1, 1)))
        # ******* diffusion part *******
        config_path = '/home/user/Documents/codes/FaceGAN/configs/restorers/GDPN/GDPN_ffhq_16xmod.py'
        cfg = Config.fromfile(config_path)
        self.diffusion = cond_diffusion.Diffusion(scale=cfg.dscale)
        # ******* refiner part ******
        # fix weights
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.fusion_diffusion.requires_grad_(False)
        self.fusion_out.requires_grad_(False)
        self.fusion_skip.requires_grad_(False)
        # ***************************
        self.refiner = nn.ModuleList()
        self.refiner.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    img_channels, rrdb_channels, num_blocks=num_rrdbs),
                nn.Conv2d(
                    rrdb_channels, channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        for res in encoder_res:
            in_channels = channels[res]
            if res > 4:
                out_channels = channels[res // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, num_styles * style_channels))
            self.refiner.append(block)

        self.fusion_outR = nn.ModuleList()
        self.fusion_skipR = nn.ModuleList()
        for res in encoder_res[::-1]:
            num_channels = channels[res]
            self.fusion_outR.append(
                nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True))
            self.fusion_skipR.append(
                nn.Conv2d(num_channels + 3, 3, 3, 1, 1, bias=True))


    def forward(self, lq):
        diffusion_recon = self.diffusion.forward(F.interpolate(lq,(256,256), mode='bicubic'))
        dfeat = diffusion_recon
        if lq.shape[-1] != 64:
            lq = F.interpolate(lq, (64, 64), mode='bicubic')
        
        # print('lq',lq.shape,'diffusion',diffusion_recon.shape)
        diffuse_encoder_features = []
        for dblock in self.encoder_diffusion:
            dfeat = dblock(dfeat)
            diffuse_encoder_features.append(dfeat)
        diffuse_encoder_features = diffuse_encoder_features[::-1]      # large to small -> small to large
        ###########################
        df_dict = {}
        for df in diffuse_encoder_features:
            # print(df.shape)
            df_dict[df.shape] = df
        ################
        feat = lq
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]      # large to small -> small to large
        latent = encoder_features[0].view(lq.size(0), -1, self.style_channels)
        # print(latent)
        encoder_features = encoder_features[1:]        # ignore 4x4

        # generator
        injected_noise = [
            getattr(self.generator, f'injected_noise_{i}')
            for i in range(self.generator.num_injected_noises)
        ]
        # 4x4 stage
        out = self.generator.constant_input(latent)
        out = self.generator.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.generator.to_rgb1(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher res
        generator_features = []
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.generator.convs[::2], self.generator.convs[1::2],
                injected_noise[1::2], injected_noise[2::2],
                self.generator.to_rgbs):

            # feature fusion by channel-wise concatenation
            if out.size(2) <= self.in_size:
                fusion_index = (_index - 1) // 2
                feat = encoder_features[fusion_index]
                # *********** -------- simple mix diffusion and encoder feat -------- **********
                feat_shape = feat.shape
                # verion1 addition
                # feat = feat + df_dict[feat_shape]
                # version2 concatenation
                feat = torch.cat([feat,df_dict[feat_shape]], dim=1)
                feat = self.fusion_diffusion[fusion_index](feat)
                ##############################
                # print('fusion group:',feat.shape) 4x4,8x8,16x16,64x64
                out = torch.cat([out, feat], dim=1)
                out = self.fusion_out[fusion_index](out)

                skip = torch.cat([skip, feat], dim=1)
                skip = self.fusion_skip[fusion_index](skip)

            # original StyleGAN operations
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)              # skip is used to monitor the recon

            # store features for decoder
            if out.size(2) > self.in_size:
                generator_features.append(out)
            _index += 2

        # decoder
        # print(skip.shape)    1x3x1024,1024

        hr = encoder_features[-1]   # [1, 512, 64, 64]
        # ********** mix with diffusion ********
        hr = hr + df_dict[encoder_features[-1].shape]
        # *************************************
        # print('HR before decoder',hr.shape)
        for i, block in enumerate(self.decoder):
            if i > 0:
                # ********** mix with diffusion ********
                if hr.shape in df_dict.keys():
                    # print(hr.shape)
                    hr = hr + df_dict[hr.shape]
                # *************************************
                hr = torch.cat([hr, generator_features[i - 1]], dim=1)
            hr = block(hr)
        # *****************Refine Stage********************
        hr_v0 = hr
        bp_hr = F.interpolate(hr,(64,64), mode='bicubic')
        feat = bp_hr
        refiner_features = []
        for block in self.refiner:
            feat = block(feat)
            refiner_features.append(feat)
        refiner_features = refiner_features[::-1]      # large to small -> small to large
        latent = refiner_features[0].view(lq.size(0), -1, self.style_channels)
        # print(latent)
        refiner_features = refiner_features[1:]        # ignore 4x4

        # generator
        injected_noise = [
            getattr(self.generator, f'injected_noise_{i}')
            for i in range(self.generator.num_injected_noises)
        ]
        # 4x4 stage
        out = self.generator.constant_input(latent)
        out = self.generator.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.generator.to_rgb1(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher res
        generator_features = []
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.generator.convs[::2], self.generator.convs[1::2],
                injected_noise[1::2], injected_noise[2::2],
                self.generator.to_rgbs):

            # feature fusion by channel-wise concatenation
            if out.size(2) <= self.in_size:
                fusion_index = (_index - 1) // 2
                feat = refiner_features[fusion_index]
                out = torch.cat([out, feat], dim=1)
                out = self.fusion_outR[fusion_index](out)
                skip = torch.cat([skip, feat], dim=1)
                skip = self.fusion_skipR[fusion_index](skip)

            # original StyleGAN operations
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)              # skip is used to monitor the recon

            # store features for decoder
            if out.size(2) > self.in_size:
                generator_features.append(out)
            _index += 2
        hr = skip
        return hr

    def init_weights(self, pretrained=None, strict=True):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

class RRDBFeatureExtractor(nn.Module):
    def __init__(self,
                 in_channels=3,
                 mid_channels=64,
                 num_blocks=23,
                 growth_channels=32):

        super().__init__()

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = make_layer(
            RRDB,
            num_blocks,
            mid_channels=mid_channels,
            growth_channels=growth_channels)
        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        return feat + self.conv_body(self.body(feat))
