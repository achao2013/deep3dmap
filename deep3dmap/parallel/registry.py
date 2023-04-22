# Copyright (c) achao2013. All rights reserved.
from torch.nn.parallel import DataParallel, DistributedDataParallel

from ..core.utils import Registry

MODULE_WRAPPERS = Registry('module wrapper')
MODULE_WRAPPERS.register_module(module=DataParallel)
MODULE_WRAPPERS.register_module(module=DistributedDataParallel)
