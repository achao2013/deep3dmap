import os
import math
from copy import deepcopy
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
import torch.distributed as dist
import torchvision
import numpy as np
import scipy.io as sio
from ..builder import MODELS, build_backbone
from collections import OrderedDict
from deep3dmap.models.frameworks.custom import CustomFramework
from deep3dmap.datasets.pipelines.formating import to_tensor
from deep3dmap.core.utils.device_transfer import to_cuda
from deep3dmap.core.utils.fileio import read_obj
from deep3dmap.core.renderer.renderer_pt3d import Pt3dRenderer
from ..builder import BACKBONES, build_backbone



@MODELS.register_module()
class gnerf(CustomFramework):
    def __init__(self, model_cfgs, train_cfg=None, test_cfg=None):
        super(gnerf, self).__init__()

        for network_name in self.network_names:
            network = getattr(self, network_name)
            network = network.cuda()
        
        self.find_unused_parameters=model_cfgs.get('find_unused_parameters', False)
        # put model on gpus
        if self.distributed and self.share_weight:
            for net_name in self.network_names:
                if self.distributed: # distributed
                    find_unused_parameters = model_cfgs.get('find_unused_parameters', False)
                    #setattr(self, net_name, MMDistributedDataParallel(getattr(self, net_name),
                    setattr(self, net_name, DDP(getattr(self, net_name),
                                            device_ids=[torch.cuda.current_device()],
                                            broadcast_buffers=False,
                                            find_unused_parameters=find_unused_parameters))
                else:
                    #model.cuda(cfg.gpu_ids[0])
                    setattr(self, net_name, DP(
                        getattr(self, net_name), device_ids=model_cfgs.gpu_ids))
    def name2net(self,name):
        return getattr(self, name)
    def optseq2netnames(self,optseq):
        if optseq=='generator_trainstep':
            self.cur_netnames=['generator']
        return self.cur_netnames
    def setup_optimize_sequences(self,state):
        if state=='A':
            self.optimize_sequences=['generator_trainstep','discriminator_trainstep','inversion_net_trainstep',
            'training_pose_regularization','training_pose_regularization']
        elif state=='ABAB':
            self.optimize_sequences=['generator_trainstep','discriminator_trainstep','inversion_net_trainstep',
            'training_pose_regularization','training_pose_regularization','training_refine_step','val_refine_step']
        elif state=='B':
            self.optimize_sequences=['training_refine_step','val_refine_step']
        else:
            assert False,'model state error'
    def train_step(self,data, state, optimize_seq):
        if optimize_seq=='generator_trainstep':
            generator=self.name2net('generator')
