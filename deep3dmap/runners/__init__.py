#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: achao
# File Name: __init__.py
# Description:
"""

from .base_module import BaseModule, ModuleList, Sequential
from .base_runner import BaseRunner
from .builder import RUNNERS, build_runner
from .checkpoint import (CheckpointLoader, _load_checkpoint,
                         _load_checkpoint_with_prefix, load_checkpoint,
                         load_state_dict, save_checkpoint, weights_to_cpu)
from .default_constructor import DefaultRunnerConstructor
from .dist_utils import (allreduce_grads, allreduce_params, get_dist_info,
                         init_dist, master_only)
from .epoch_based_runner import EpochBasedRunner
from .gan2shape_runner import Gan2ShapeRunner
from .state_machine_runner import StateMachineRunner
from .hooks import (HOOKS, CheckpointHook, ClosureHook, DistEvalHook,
                    DistSamplerSeedHook, DvcliveLoggerHook, EMAHook, EvalHook,
                    GradientCumulativeOptimizerHook, Hook, IterTimerHook,
                    LoggerHook, LrUpdaterHook, MlflowLoggerHook,
                    NeptuneLoggerHook, OptimizerHook, PaviLoggerHook,
                    SyncBuffersHook, TensorboardLoggerHook, TextLoggerHook,
                    WandbLoggerHook)
from .iter_based_runner import IterBasedRunner, IterLoader
from .log_buffer import LogBuffer
from .optimizer import (OPTIMIZER_BUILDERS, OPTIMIZERS,
                        DefaultOptimizerConstructor, build_optimizer,
                        build_optimizer_constructor)
from .priority import Priority, get_priority
from .utils import get_host_info, get_time_str, obj_from_dict, set_random_seed
from .forward_engine import single_gpu_test, multi_gpu_test

__all__ = [
    'BaseRunner', 'EpochBasedRunner', 'IterBasedRunner', 'LogBuffer',
    'HOOKS', 'Hook', 'CheckpointHook', 'ClosureHook', 'LrUpdaterHook',
    'OptimizerHook', 'IterTimerHook', 'DistSamplerSeedHook', 'LoggerHook',
    'PaviLoggerHook', 'TextLoggerHook', 'TensorboardLoggerHook',
    'NeptuneLoggerHook', 'WandbLoggerHook', 'MlflowLoggerHook',
    'DvcliveLoggerHook', '_load_checkpoint', 'load_state_dict',
    'load_checkpoint', 'weights_to_cpu', 'save_checkpoint', 'Priority',
    'get_priority', 'get_host_info', 'get_time_str', 'obj_from_dict',
    'init_dist', 'get_dist_info', 'master_only', 'OPTIMIZER_BUILDERS',
    'OPTIMIZERS', 'DefaultOptimizerConstructor', 'build_optimizer',
    'build_optimizer_constructor', 'IterLoader', 'set_random_seed',
    'SyncBuffersHook', 'EMAHook', 'build_runner', 'RUNNERS', 'allreduce_grads',
    'allreduce_params', 'CheckpointLoader', 'BaseModule',
    '_load_checkpoint_with_prefix', 'EvalHook', 'DistEvalHook', 'Sequential',
    'ModuleList', 'GradientCumulativeOptimizerHook',
    'DefaultRunnerConstructor', 'Gan2ShapeRunner', 'single_gpu_test', 'multi_gpu_test',
    'StateMachineRunner'
]
