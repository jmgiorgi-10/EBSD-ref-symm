import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.slerp3 import quat_upsampling, quat_upsampling_symm3

def default_pad(input, pad, mode='replicate'):
    return nn.functional.pad(
        input, pad, mode
    )

def transp_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        #import pdb; pdb.set_trace()
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Slerp(nn.Module):
    def __init__(self, scale):
        self.scale = scale
        super(Slerp, self).__init__()
    
    def forward(self, x):
        batch_num = x.shape[0]
        # inherits 4 quaternion channels, after the 2d transposed convolution
        device=torch.device('cuda:0')
        sr = torch.zeros(batch_num, 4, self.scale*x.shape[2]-(self.scale-1), self.scale*x.shape[3]-(self.scale-1),device=device)

        for i in range(batch_num):
            upsampled = quat_upsampling_symm3(x[i,...], self.scale)
            sr[i,...] = upsampled

        sr = default_pad(sr, [0, 3, 0, 3], 'replicate') 

        return sr

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        #import pdb; pdb.set_trace()
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                #m.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
                #m.append(conv(n_feat, n_feat, 3, bias))

                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                #m.append(conv(n_feat, 2 * n_feat, 3, bias))
                # m.append(conv(2 * n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

