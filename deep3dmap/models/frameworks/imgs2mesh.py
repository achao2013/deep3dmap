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
from ..builder import MODELS, build_backbone
from deep3dmap.models.frameworks.custom import CustomFramework
from deep3dmap.datasets.pipelines.formating import to_tensor
from deep3dmap.core.utils.device_transfer import to_cuda

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
        self.template_normals=
        self.template_normals=to_cuda(to_tensor(self.template_normals))
        self.verts_uvs = np.load(model_cfgs.get('uvs_path', "data/uvs.npy"))
        self.verts_uvs = to_cuda(to_tensor(self.verts_uvs))