# Copyright (c) achao2013. All rights reserved.
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from deep3dmap.runners.base_module import BaseModule
from ..builder import BACKBONES

@BACKBONES.register_module()
class Encoder(BaseModule):
    def __init__(self, cin, cout, size=64, nf=64, activation=nn.Tanh):
        super(Encoder, self).__init__()
        extra = int(np.log2(size) - 6)
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.ReLU(inplace=True)]
        for i in range(extra):
            nf *= 2
            network += [
                nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
                nn.ReLU(inplace=True)]
        network += [
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, cout, kernel_size=1, stride=1, padding=0, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0),-1)


class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockDown, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.activation(x)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        if self.in_channels != self.out_channels:
            x = self.conv_sc(x)
        return x + h

@BACKBONES.register_module()
class ResEncoder(BaseModule):
    def __init__(self, cin, cout, size=128, nf=16, activation=None):
        super(ResEncoder, self).__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            ResBlockDown(nf, nf*2),
            ResBlockDown(nf*2, nf*4),
            ResBlockDown(nf*4, nf*8)]
        extra = int(np.log2(size) - 6)
        for i in range(extra):
            nf *= 2
            network += [ResBlockDown(nf*4, nf*8)]
        network += [
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*16, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*16, cout, kernel_size=1, stride=1, padding=0, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0),-1)