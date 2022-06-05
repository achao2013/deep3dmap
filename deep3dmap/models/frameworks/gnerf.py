import os
import math
from copy import deepcopy
from sys import set_asyncgen_hooks
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
from deep3dmap.parallel import MMDataParallel, MMDistributedDataParallel

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
                             radius=model_cfgs.radius, look_at_origin=model_cfgs.look_at_origin, ndc=model_cfgs.ndc)

        self.network_names=[]
        self.generator=GNeRF(
            ray_sampler=self.ray_sampler, xyz_freq=model_cfgs.xyz_freq, dir_freq=model_cfgs.xyz_freq, fc_depth=model_cfgs.fc_depth,
            fc_dim=model_cfgs.fc_dim, chunk=model_cfgs.chunk, white_back=model_cfgs.white_back)
        self.network_names.append('generator')
        self.discriminator = Discriminator(
            conditional=model_cfgs.conditional, policy=model_cfgs.policy, ndf=model_cfgs.ndf, imsize=model_cfgs.patch_size)
        self.network_names.append('discriminator')
        self.inv_net = InversionNet(imsize=model_cfgs.inv_size, pose_mode=model_cfgs.pose_mode)
        self.network_names.append('inv_net')
        self.pose_mode=model_cfgs.pose_mode

        
        self.distributed=model_cfgs.get('distributed',True)

        for net_name in self.network_names:
            net=self.name2net(net_name)
            if self.distributed:
                find_unused_parameters = model_cfgs.get('find_unused_parameters', False)
                # Sets the `find_unused_parameters` parameter in
                # torch.nn.parallel.DistributedDataParallel
                
                setattr(self, net_name, MMDistributedDataParallel(
                    net.cuda(),
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False,
                    find_unused_parameters=find_unused_parameters))
            else:
                setattr(self, net_name, MMDataParallel(
                    net.cuda(model_cfgs.gpu_ids[0]), device_ids=model_cfgs.gpu_ids))
    def set_info_from_datasets(self,datasets):
        self.ray_sampler.set_start_intrinsics(datasets[0].intrinsics.clone().detach().cuda())
        self.train_pose_params = PoseParameters(
            length=len(datasets[0]), pose_mode=self.pose_mode, data=datasets[0].name)
        self.val_pose_params = PoseParameters(
            length=len(datasets[1]), pose_mode=self.pose_mode, data=datasets[1].name)
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
        if self.cfg.progressvie_training:
            img_real = self.progressvie_training(img_real)
            val_imgs = self.progressvie_training(val_imgs_raw)

            self.generator.ray_sampler.update_intrinsic(self.img_wh_curr / self.img_wh_end)

        if self.cfg.decrease_noise:
            self.generator.decrease_noise(self.it)

        self.dynamic_patch_sampler.iterations = self.it
        if optimize_seq=='generator_trainstep':
            generator=self.name2net('generator')
