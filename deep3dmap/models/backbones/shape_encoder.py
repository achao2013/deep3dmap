# Copyright (c) achao2013. All rights reserved.
from turtle import forward
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
from deep3dmap.runners.base_module import BaseModule
from ..builder import BACKBONES, build_backbone

def reset_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, 0.0, 0.0001)
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.normal(m.weight, 1.0, 0.02)
            nn.init.constant(m.bias, 0)

@BACKBONES.register_module()
class Shape3dmmEncoder(BaseModule):
    def __init__(self, init_cfg=None, net_name="Vgg"):
        super().__init__(init_cfg)
        self.featChannel = 512
        self.feat_net=build_backbone(dict(type=net_name))
        self.fc_3dmm = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.featChannel, 256*2)),
            ('relu1', nn.ReLU(True)),
            ('fc2', nn.Linear(256*2, 228))]))
        
        self.fc_pose = nn.Sequential(OrderedDict([
           ('fc3', nn.Linear(512, 256)),
           ('relu2', nn.ReLU(True)),
           ('fc4', nn.Linear(256, 7))]))
        reset_params(self.fc_3dmm)
        reset_params(self.fc_pose)
    def forward(self, img):
        feat=self.feat_net(img)
        pose = self.fc_pose(feat)
        param = self.fc_3dmm(feat)
        out = torch.cat([param, pose], dim=1)
        return out