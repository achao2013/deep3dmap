#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: achao
# File Name: dist_utils.py
# Description:
"""
import functools
import os
import subprocess
from collections import OrderedDict

import torch
import torch.multiprocessing as mp
from torch import distributed as dist
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)

def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size
