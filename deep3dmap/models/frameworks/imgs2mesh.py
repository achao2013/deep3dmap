import os
import math
from copy import deepcopy

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
from deep3dmap.models.frameworks.custom import CustomFramework
from deep3dmap.datasets.pipelines.formating import to_tensor
from deep3dmap.core.utils.device_transfer import to_cuda
from deep3dmap.core.utils.fileio import read_obj
from deep3dmap.core.renderer.renderer_pt3d import Pt3dRenderer

@MODELS.register_module()
class imgs2mesh(CustomFramework):
    def __init__(self, model_cfgs, train_cfg=None, test_cfg=None):
        super(imgs2mesh, self).__init__()
        # basic parameters
        self.model_name = model_cfgs.get('model_name', self.__class__.__name__)
        self.checkpoint_dir = model_cfgs.get('checkpoint_dir', 'results')
        self.distributed = model_cfgs.get('distributed')
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.use_sampling=model_cfgs.get('use_sampling', False)
        if self.use_sampling:
            v,f,normals = read_obj(model_cfgs.get('template_normal_path',"data/template_normal.obj"))
            if np.mean(normals[:,2])<0:
                normals = -normals
            self.template_normals=np.array(normals)
            self.template_normals=to_cuda(to_tensor(self.template_normals))
        self.template_uvs = np.load(model_cfgs.get('template_uvs_path', "data/uvs.npy"))
        self.template_uvs = to_cuda(to_tensor(self.verts_uvs))
        self.model_param = sio.loadmat(model_cfgs.get('model_param_path','data/Model_Shape.mat'))
        self.lm68idx=self.model_param['keypoints'][0].astype(np.int32)
        self.lookview=to_cuda(to_tensor(np.array([0,0,1])))
        self.renderer=Pt3dRenderer()

