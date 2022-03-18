import os
import os.path as osp
import platform
import shutil
import time
import warnings
from deep3dmap.core import utils
from deep3dmap.runners.optimizer import build_optimizer
import torch.distributed as dist

import torch

import deep3dmap.core as core
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info


@RUNNERS.register_module()
class CustomRunner(BaseRunner):
    """Custom Runner.

    This runner train models epoch by epoch.
    """
    def __init__(self, runner_cfgs, model,work_dir=None,
            logger=None,
            meta=None):
        super().__init__(model, None, work_dir, logger, meta, None, None)