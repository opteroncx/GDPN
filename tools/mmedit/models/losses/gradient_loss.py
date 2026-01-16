# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .pixelwise_loss import l1_loss
import numpy as np

_reduction_modes = ['none', 'mean', 'sum']

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

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError

class L1_Wavelet_Loss(nn.Module):
    def __init__(self):
        super(L1_Wavelet_Loss, self).__init__()
        self.wave = WavePool(3)
        self.eps = 1e-6

    def forward(self, X, Y):
        Y_outs = self.wave(Y)
        Yc = torch.cat(Y_outs,1)
        X = self.wave(X)
        Xc = torch.cat(X,1)
        loss = F.l1_loss(Xc, Yc)
        return loss

class AdvancedSobel(nn.Module):
    def __init__(self):
        super(AdvancedSobel, self).__init__()
        self.conv_op_x = nn.Conv2d(3, 1, 3,stride=1, padding=1, bias=False)
        self.conv_op_y = nn.Conv2d(3, 1, 3,stride=1, padding=1, bias=False)
        self.conv_op_a45 = nn.Conv2d(3, 1, 3,stride=1, padding=1, bias=False)
        self.conv_op_a135 = nn.Conv2d(3, 1, 3,stride=1, padding=1, bias=False)
        sobel_kernel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype='float32')
        sobel_kernel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype='float32')
        sobel_kernel_a45 = np.array([[[0, 1, 2], [-1, 0, 1], [-2, -1, -0]],
                                   [[0, 1, 2], [-1, 0, 1], [-2, -1, -0]],
                                   [[0, 1, 2], [-1, 0, 1], [-2, -1, -0]]], dtype='float32')
        sobel_kernel_a135 = np.array([[[2, 1, 0], [1, 0, -1], [0, -1, -2]],
                                   [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
                                   [[2, 1, 0], [1, 0, -1], [0, -1, -2]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))
        sobel_kernel_a45 = sobel_kernel_a45.reshape((1, 3, 3, 3))
        sobel_kernel_a135 = sobel_kernel_a135.reshape((1, 3, 3, 3))

        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y)
        self.conv_op_a45.weight.data = torch.from_numpy(sobel_kernel_a45)
        self.conv_op_a135.weight.data = torch.from_numpy(sobel_kernel_a135)
        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False
        self.conv_op_a45.weight.requires_grad = False
        self.conv_op_a135.weight.requires_grad = False

    def forward(self, x):
        edge_Y_x = self.conv_op_x(x)
        edge_Y_y = self.conv_op_y(x)
        # edge_Y = torch.abs(edge_Y_x) + torch.abs(edge_Y_y)
        return edge_Y_x, edge_Y_y



class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        # self.device = device
        self.conv_op_x = nn.Conv2d(3, 1, 3,stride=1, padding=1, bias=False)
        self.conv_op_y = nn.Conv2d(3, 1, 3,stride=1, padding=1, bias=False)

        sobel_kernel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype='float32')
        sobel_kernel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))

        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y)
        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False

    def forward(self, x):
        edge_Y_x = self.conv_op_x(x)
        edge_Y_y = self.conv_op_y(x)
        # edge_Y = torch.abs(edge_Y_x) + torch.abs(edge_Y_y)
        return edge_Y_x,edge_Y_y

@LOSSES.register_module()
class GradientLoss(nn.Module):
    """Gradient loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        edge_type = 'ASL'
        if edge_type == 'Sobel':
            self.sobel = Sobel()
        elif edge_type == 'ASL':
            self.sobel = AdvancedSobel()

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
        #                    [1, 0, -1]]).view(1, 1, 3, 3).to(target)
        # ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
        #                    [-1, -2, -1]]).view(1, 1, 3, 3).to(target)

        # pred_grad_x = F.conv2d(pred, kx, padding=1)
        # pred_grad_y = F.conv2d(pred, ky, padding=1)
        # target_grad_x = F.conv2d(target, kx, padding=1)
        # target_grad_y = F.conv2d(target, ky, padding=1)

        # loss = (
        #     l1_loss(
        #         pred_grad_x, target_grad_x, weight, reduction=self.reduction) +
        #     l1_loss(
        #         pred_grad_y, target_grad_y, weight, reduction=self.reduction))
        pred_grad_x,pred_grad_y = self.sobel(pred)
        target_grad_x,target_grad_y = self.sobel(target)
        loss = (
            l1_loss(
                pred_grad_x, target_grad_x, weight, reduction=self.reduction) +
            l1_loss(
                pred_grad_y, target_grad_y, weight, reduction=self.reduction))

        return loss * self.loss_weight
