# Copyright (c) achao2013. All rights reserved.
#coding=utf-8
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from torch.nn.parameter import Parameter
import torch.utils.data
from torch.nn import functional as F

from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple

from ..builder import BACKBONES


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    # 函数中padding参数可以无视，实际实现的是padding=same的效果
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.zeros(
                (in_channels, out_channels // groups, *kernel_size)))
        else:
            self.weight = Parameter(torch.zeros(
                (out_channels, in_channels // groups, *kernel_size)))
        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class Conv2d(_ConvNd): 

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    # 修改这里的实现函数
    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, outplanes, kernel_size=4, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(inplanes, outplanes//2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes//2)
        self.conv2 = Conv2d(outplanes//2, outplanes//2, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes//2)
        self.conv3 = Conv2d(outplanes//2, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.bn3(out)
        out = self.relu(out)

        return out

@BACKBONES.register_module()
class resfcn256_std(nn.Module):
    def __init__(self, resolution_inp = 256, resolution_op = 256, channel = 3, name = 'resfcn256'):
        super(resfcn256_std, self).__init__()
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.inplanes = 16

        self.conv1 = Conv2d(3, 16, kernel_size=4, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(Bottleneck, 32, 2, stride=2)
        self.layer2 = self._make_layer(Bottleneck, 64, 2, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 128, 2, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 256, 2, stride=2)
        self.layer5 = self._make_layer(Bottleneck, 512, 2, stride=2)
        decoder = [
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.ConvTranspose2d(512,512, 4,1,2, bias=False), # B x 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512,256, 4,2,1, bias=False), # B x 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.ConvTranspose2d(256,256, 4,1,2, bias=False), # B x 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.ConvTranspose2d(256,256, 4,1,2, bias=False), # B x 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128, 4,2,1, bias=False), # B x 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.ConvTranspose2d(128,128, 4,1,2, bias=False), # B x 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.ConvTranspose2d(128,128, 4,1,2, bias=False), # B x 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64, 4,2,1, bias=False),  # B x 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.ConvTranspose2d(64,64, 4,1,2, bias=False),   # B x 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.ConvTranspose2d(64,64, 4,1,2, bias=False),   # B x 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,32, 4,2,1, bias=False),   # B x 32 x 128 x 128
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.ConvTranspose2d(32,32, 4,1,2, bias=False),   # B x 32 x 128 x 128
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,16, 4,2,1, bias=False),   # B x 16 x 256 x 256
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.ConvTranspose2d(16,16, 4,1,2, bias=False),   # B x 16 x 256 x 256
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.ConvTranspose2d(16,3, 4,1,2, bias=False),    # B x 3 x 256 x 256
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.ConvTranspose2d(3,3, 4,1,2, bias=False),     # B x 3 x 256 x 256
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.ConvTranspose2d(3,3, 4,1,2, bias=False),     # B x 3 x 256 x 256
            nn.Sigmoid(),
        ]
        self.decoder=nn.Sequential(*decoder)
        self.apply(self.weights_init)



    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, 4, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.decoder(x)

        return x


