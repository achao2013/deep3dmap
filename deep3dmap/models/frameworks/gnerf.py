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

from deep3dmap.models.modulars.gnerf import GNeRF
from deep3dmap.models.modulars.dynamic_patch_discriminator import Discriminator
from deep3dmap.core.renderer.samples.patch_sampler import FlexPatchSampler,FullImageSampler,RescalePatchSampler 
from deep3dmap.core.renderer.samples.ray_sampler import RaySampler
from deep3dmap.models.modulars.embeddings import PoseParameters
from deep3dmap.models.modulars.inversion_net import InversionNet

@MODELS.register_module()
class GanNerf(CustomFramework):
    def __init__(self, model_cfgs, train_cfg=None, test_cfg=None):
        super(GanNerf, self).__init__()
        
        self.dynamic_patch_sampler = FlexPatchSampler(
            random_scale=model_cfgs.random_scale,
            min_scale=model_cfgs.min_scale,
            max_scale=model_cfgs.max_scale,
            scale_anneal=model_cfgs.scale_anneal,
        )

        self.static_patch_sampler = RescalePatchSampler()

        self.full_img_sampler = FullImageSampler()

        self.ray_sampler = RaySampler(near=model_cfgs.near, far=model_cfgs.far, azim_range=model_cfgs.azim_range, elev_range=model_cfgs.elev_range,
                             radius=model_cfgs.radius, look_at_origin=model_cfgs.look_at_origin, ndc=model_cfgs.ndc,
                             intrinsics=train_loader.dataset.intrinsics.clone().detach())
        self.generator=GNeRF(
            ray_sampler=self.ray_sampler, xyz_freq=model_cfgs.xyz_freq, dir_freq=model_cfgs.xyz_freq, fc_depth=model_cfgs.fc_depth,
            fc_dim=model_cfgs.fc_dim, chunk=model_cfgs.chunk, white_back=model_cfgs.white_back)
        self.discriminator = Discriminator(
            conditional=model_cfgs.conditional, policy=model_cfgs.policy, ndf=model_cfgs.ndf, imsize=model_cfgs.patch_size)
        self.inv_net = InversionNet(imsize=model_cfgs.inv_size, pose_mode=model_cfgs.pose_mode)

        self.train_pose_params = PoseParameters(
            length=len(train_loader.dataset), pose_mode=model_cfgs.pose_mode, data=model_cfgs.data)
        self.val_pose_params = PoseParameters(
            length=len(eval_loader.dataset), pose_mode=model_cfgs.pose_mode, data=model_cfgs.data)

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
