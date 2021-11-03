#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: achao
# File Name: gan2shape_runner.py
# Description:
"""

import os
import os.path as osp
import platform
import shutil
import time
import warnings
from deep3dmap.core import utils
from deep3dmap.runners import build_optimizer
import torch.distributed as dist

import torch

import deep3dmap.core as core
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info


@RUNNERS.register_module()
class Gan2ShapeRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    def __init__(self, runner_cfgs, model):
        super(Gan2ShapeRunner).__init__(model)
        # basic parameters
        self.distributed = runner_cfgs.get('distributed')
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.checkpoint_dir = runner_cfgs.get('checkpoint_dir', 'results')
        self.save_checkpoint_freq = runner_cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = runner_cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints
        self.use_logger = runner_cfgs.get('use_logger', True)
        self.log_freq = runner_cfgs.get('log_freq', 1000)
        self.make_metrics = lambda m=None, mode='moving': meters.StandardMetrics(m, mode)
        #self.model = model(runner_cfgs)

        self.dataset = None
        self.data_loader = None

        # functional parameters
        self.joint_train = runner_cfgs.get('joint_train', False)  # True: joint train on multiple images
        self.independent = runner_cfgs.get('independent', True)  # True: each process has a different image
        self.reset_weight = runner_cfgs.get('reset_weight', False)
        self.load_gt_depth = runner_cfgs.get('load_gt_depth', False)
        self.save_results = runner_cfgs.get('save_results', False)
        # detailed parameters
        self.num_stage = runner_cfgs.get('num_stage')
        self.stage_len_dict = runner_cfgs.get('stage_len_dict')
        self.stage_len_dict2 = runner_cfgs.get('stage_len_dict2', None)
        self.flip1_cfg = runner_cfgs.get('flip1_cfg', [False, False, False])
        self.flip3_cfg = runner_cfgs.get('flip3_cfg', [False, False, False])

        self.mode_seq = ['step1', 'step2', 'step3']
        self.current_stage = 0
        self.count = 0

        self.setup_state()

        if self.save_results and self.rank == 0:
            img_save_path = self.checkpoint_dir + '/images'
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)
        
        self.make_optimizer = lambda model: build_optimizer(model, runner_cfgs.optimizer)         
        
    def init_optimizers(self):
        self.optimizer_names = []
        if self.model.mode == 'step1':
            optimize_names = ['netA']
        elif self.model.mode == 'step2':
            optimize_names = ['netEnc']
        elif self.model.mode == 'step3':
            optimize_names = [name for name in self.model.network_names]
            optimize_names.remove('netEnc')

        for net_name in optimize_names:
            optimizer = self.make_optimizer(getattr(self.model, net_name))
            optim_name = net_name.replace('net', 'optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def backward(self):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).zero_grad()
        self.model.loss_total.backward()
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).step()

    def setup_data(self, epoch):
        if self.joint_train:
            self.dataset.setup_input(self.img_list,
                                    self.depth_list if self.load_gt_depth else None,
                                    self.latent_list)
            
        elif self.independent:
            idx = epoch * self.world_size
            self.dataset.setup_input(self.img_list[idx:idx+self.world_size],
                                    self.depth_list[idx:idx+self.world_size] if self.load_gt_depth else None,
                                    self.latent_list[idx:idx+self.world_size])
        else:
            self.dataset.setup_input(self.img_list[epoch],
                                    self.depth_list[epoch] if self.load_gt_depth else None,
                                    self.latent_list[epoch])
        self.model.setup_target(self.dataset.img_num, self.dataset.input_im, self.dataset.input_im_all, 
            self.dataset.w_path, self.dataset.latent_w, self.dataset.latent_w_all)

    def setup_mode(self):
        stage_len_dict = self.stage_len_dict if self.current_stage == 0 else self.stage_len_dict2
        if self.count >= stage_len_dict[self.model.mode]:
            if (self.independent or self.rank == 0) and self.save_results:
                if self.model.mode == 'step3':
                    self.model.save_results(self.current_stage+1)
                elif self.model.mode == 'step2' and self.current_stage == 0:
                    self.model.save_results(self.current_stage)
            if self.model.mode == 'step1' and self.joint_train:
                # collect results in step1
                self.model.step1_collect()
            if self.model.mode == 'step2':
                # collect projected samples
                self.model.step2_collect()
            if self.model.mode == self.mode_seq[-1]:  # finished a stage
                self.current_stage += 1
                if self.current_stage >= self.num_stage:
                    return -1
                self.setup_state()
            idx = self.mode_seq.index(self.model.mode)
            next_mode = self.mode_seq[(idx + 1) % len(self.mode_seq)]
            self.model.mode = next_mode
            self.model.init_optimizers()
            self.metrics.reset()
            self.count = 0
        self.count += 1
        return 1

    def setup_state(self):
        self.model.flip1 = self.flip1_cfg[self.current_stage]
        self.model.flip3 = self.flip3_cfg[self.current_stage]

    def reset_state(self):
        self.current_stage = 0
        self.count = 0
        self.model.mode = 'step1'
        if self.reset_weight:
            self.model.reset_model_weight()
        self.model.init_optimizers()
        self.model.canon_mask = None
        self.setup_state()

    def run_iter(self, data_batch, train_mode, **kwargs):
        if train_mode:
            state = self.setup_mode()
            if state < 0:
                self.metrics_all.update(m, 1)
                if self.rank == 0:
                    print(f"{'Epoch'}{self.epoch:05}/{self.metrics_all}")
                    self.save_checkpoint(self.iteration_all)
                return
            m = self.model.forward()
            self.backward()
            #outputs = self.model.train_step(data_batch, self.optimizer,
            #                                **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self.reset_state()
        self.setup_data(self.epoch)
        self.metrics.reset()

        #self._max_iters = self._max_epochs * len(self.data_loader)
        #self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            #self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            #self.call_hook('after_train_iter')
            self._iter += 1
            

        #self.call_hook('after_train_epoch')
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

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        self.model.init_optimizers()
        assert isinstance(data_loaders, list)
        assert core.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        """
            if self.joint_train:
                self._max_epochs = 1
            elif self.independent:
                self._max_epochs = len(self.img_list) // self.world_size
            else:
                self._max_epochs = len(self.img_list)
        """

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        
        self.metrics = self.make_metrics(mode='moving')
        self.metrics_all = self.make_metrics(mode='total')
        
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

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        #self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                core.utils.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
