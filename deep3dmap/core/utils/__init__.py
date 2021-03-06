#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: achao
# File Name: __init__.py
# Description:
"""
from .registry import Registry, build_from_cfg
from .env import collect_env
from .logging import get_logger, print_log, get_root_logger
from .config import Config, ConfigDict, DictAction
from .misc import (check_prerequisites, concat_list, deprecated_api_warning,
                   import_modules_from_strings, is_list_of,
                   is_method_overridden, is_seq_of, is_str, is_tuple_of,
                   iter_cast, list_cast, requires_executable, requires_package,
                   slice_list, to_1tuple, to_2tuple, to_3tuple, to_4tuple,
                   to_ntuple, tuple_cast)
from .meters import StandardMetrics
from .path import (check_file_exist, fopen, is_filepath, mkdir_or_exist,
                   scandir, symlink)
from .version_utils import digit_version, get_git_hash
from .info import (get_compiler_version, get_compiling_cuda_version)
from .flops_counter import get_model_complexity_info
from .fuse_conv_bn import fuse_conv_bn
from .sync_bn import revert_sync_batchnorm
from .weight_init import (INITIALIZERS, Caffe2XavierInit, ConstantInit,
                          KaimingInit, NormalInit, PretrainedInit,
                          TruncNormalInit, UniformInit, XavierInit,
                          bias_init_with_prob, caffe2_xavier_init,
                          constant_init, initialize, kaiming_init, normal_init,
                          trunc_normal_init, uniform_init, xavier_init)
from .progressbar import (ProgressBar, track_iter_progress,
                          track_parallel_progress, track_progress)
from .fileio import *
from .parrots_wrapper import (
        TORCH_VERSION, BuildExtension, CppExtension, CUDAExtension, DataLoader,
        PoolDataLoader, SyncBatchNorm, _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd,
        _AvgPoolNd, _BatchNorm, _ConvNd, _ConvTransposeMixin, _InstanceNorm,
        _MaxPoolNd, get_build_config, is_rocm_pytorch, _get_cuda_home)
__all__ = [
        'Config', 'ConfigDict', 'DictAction', 'is_str', 'iter_cast',
        'list_cast', 'tuple_cast', 'is_seq_of', 'is_list_of', 'is_tuple_of',
        'slice_list', 'concat_list', 'StandardMetrics', 'Registry', 'build_from_cfg',
        'get_model_complexity_info', 'bias_init_with_prob', 'caffe2_xavier_init',
        'constant_init', 'kaiming_init', 'normal_init', 'trunc_normal_init',
        'uniform_init', 'xavier_init', 'fuse_conv_bn', 'initialize',
        'INITIALIZERS', 'ConstantInit', 'XavierInit', 'NormalInit',
        'TruncNormalInit', 'UniformInit', 'KaimingInit', 'PretrainedInit',
        'Caffe2XavierInit', 'revert_sync_batchnorm', 'SyncBatchNorm',
        '_AdaptiveAvgPoolNd', '_AdaptiveMaxPoolNd', '_AvgPoolNd', '_BatchNorm',
        '_ConvNd', '_ConvTransposeMixin', '_InstanceNorm', '_MaxPoolNd',
        'get_build_config', 'BuildExtension', 'CppExtension', 'CUDAExtension',
        'DataLoader', 'PoolDataLoader', 'TORCH_VERSION','is_rocm_pytorch',
        '_get_cuda_home', 'ProgressBar',
        'track_progress', 'track_iter_progress', 'track_parallel_progress',
        ]
