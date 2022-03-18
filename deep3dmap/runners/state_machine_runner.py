#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: achao
# File Name: state_machine_runner.py
# Description:
"""

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
class StateMachineRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    def __init__(self, runner_cfgs, model, work_dir=None,
            logger=None,
            meta=None):
        super().__init__(model, None, work_dir, logger, meta, None, None)
        self.model = model

        self.datasets = [None]
        self.data_loaders = [None]
        self.state_steps = runner_cfgs.get('state_steps', {'sup':16,'sup_unsup':32})
        self.state_seq = runner_cfgs.get('state_seq', ['sup', 'sup_unsup'])
        self.state='sup'
        self.state_switch_mode = runner_cfgs.get('state_switch_mode', 'epoch_steps')
        self.state_switch_method = runner_cfgs.get('state_switch_method','once_inorder')
        self.max_epochs=runner_cfgs.get('max_epochs','32')
        self.optimizer = build_optimizer(model, runner_cfgs.optimizer)
    def state_switch(self):
        if self.state_switch_mode == 'epoch_steps':
            if self.epoch >= self.state_steps[self.state]:
                if self.state_switch_method=='once_inorder':
                    idx = self.state_seq.index(self.mode)
                    if idx<len(self.state_seq)-1:
                        self.state=self.state_seq[idx+1]
                elif self.state_switch_method=='loop_inorder':
                    idx = self.state_seq.index(self.mode)
                    next_state = self.state_seq[(idx + 1) % len(self.state_seq)]
                    self.state = next_state

    def run_iter(self, data_batch, train_mode, **kwargs):
        if train_mode:
            preds = self.model.forward(data_batch, **kwargs)
            outputs = self.model.cal_loss(preds, data_batch, **kwargs)
        else:
            preds = self.model.forward(data_batch, **kwargs)
            outputs = self.model.cal_loss(preds, data_batch, **kwargs)

        if not isinstance(outputs, dict):
            raise TypeError('"model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, use_data_loaders, **kwargs):
        self.model.train()
        self.mode = 'train'
        
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        state_idx=0
        
        if use_data_loaders:
            for i in range(len(self.data_loaders)):
                if self.state==self.data_loaders[i].dataset.state:
                    state_idx=i
            for i, data_batch in enumerate(self.data_loaders[state_idx]):
                self._inner_iter = i
                self.call_hook('before_train_iter')
                self.run_iter(data_batch, train_mode=True, **kwargs)           
                #print('after run_iter and before after_train_iter')
                self.call_hook('after_train_iter')
                self._iter += 1
        else:
            for i in range(len(self.datasets)):
                if self.state==self.datasets[i].state:
                    state_idx=i
            for i in range(self.datasets[state_idx].iter_size):
                data_batch=self.datasets[state_idx].get()
                self._inner_iter = i
                self.call_hook('before_train_iter')
                self.run_iter(data_batch, train_mode=True, **kwargs)           
                #print('after run_iter and before after_train_iter')
                self.call_hook('after_train_iter')
                self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, use_data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            use_data_loaders (bool): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        
        assert core.is_list_of(workflow, tuple)
            
        
        
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs



        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                if use_data_loaders:
                    self._max_iters = self._max_epochs * len(self.data_loaders[i])
                else:
                    self._max_iters = self._max_epochs * len(self.datasets[i])
                break
                

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        
        
        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                if mode == 'train':
                    self.state_switch()
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(use_data_loaders,  **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')