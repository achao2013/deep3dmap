# Copyright (c) achao2013. All rights reserved.
#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: achao
# File Name: celeba.py
# Description:
"""
from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose
from torchvision import transforms
import torch
import torch.distributed as dist
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
from PIL import Image

import numpy as np
from deep3dmap.core.utils import print_log

@DATASETS.register_module()
class CelebaDataset(CustomDataset):
    CLASSES = ('face')

    def __init__(self,
                 img_list_path,
                 img_root,
                 latent_root,
                 distributed,
                 image_size,
                 load_gt_depth=False,
                 test_mode=False,
                 joint_train=False,
                 independent=True,
                 pipeline=None,
                 crop=None,
                 filter_empty_gt=True):
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.joint_train=joint_train
        self.load_gt_depth=load_gt_depth
        self.independent=independent
        self.crop=crop
        self.image_size = image_size
        self.epoch=0
        self.distributed = distributed
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1

        img_list_file = open(img_list_path)
        self.img_list, self.depth_list, self.latent_list = [], [], []
        for line in img_list_file.readlines():
            img_name = line.split()[0]
            img_path = osp.join(img_root, img_name)
            latent_path = osp.join(latent_root, img_name.replace('.png', '.pt'))
            self.img_list.append(img_path)
            if self.load_gt_depth:
                self.depth_list.append(img_path.replace('image', 'depth'))
            self.latent_list.append(latent_path)
        if self.independent:
            assert len(self.img_list) % self.world_size == 0

        self.image_path=None
        self.gt_depth_path=None
        self.w_path=None

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.img_list)
    def load_data(self):
        transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )

        def load_depth(depth_path):
            depth_gt = Image.open(depth_path)
            if self.crop is not None:
                depth_gt = transforms.CenterCrop(self.crop)(depth_gt)
            depth_gt = transform(depth_gt).cuda()
            depth_gt = (1 - depth_gt) * 2 - 1
            depth_gt = self.depth_rescaler(depth_gt)
            return depth_gt

        def load_image(image_path):
            image = Image.open(image_path)
            self.origin_size = image.size[0]  # we assume h=w
            if self.crop is not None:
                image = transforms.CenterCrop(self.crop)(image)
            image = transform(image).unsqueeze(0).cuda()
            image = image * 2 - 1
            return image

        if self.joint_train:
            assert type(self.image_path) is list
            self.input_im_all = []
            if self.load_gt_depth:
                self.depth_gt_all = []
            self.img_num = len(self.image_path)
            #assert self.collect_iters >= self.img_num
            print("Loading images...")
            for i in range(self.img_num):
                image_path = self.image_path[i]
                input_im = load_image(image_path)
                self.input_im_all.append(input_im.cpu())
                if self.load_gt_depth:
                    depth_path = self.gt_depth_path[i]
                    depth_gt = load_depth(depth_path)
                    self.depth_gt_all.append(depth_gt.cpu())
            self.input_im = self.input_im_all[0].cuda()
            if self.load_gt_depth:
                self.depth_gt = self.depth_gt_all[0].cuda()
            # img_idx is used to track the index of current image
            self.img_idx = 0
            self.idx_perm = torch.LongTensor(list(range(self.img_num)))
        else:
            self.img_num = len(self.image_path)
            if type(self.image_path) is list:
                assert len(self.image_path) == self.world_size
                
                self.image_path = self.image_path[self.rank]
                if self.load_gt_depth:
                    self.gt_depth_path = self.gt_depth_path[self.rank]
            print("Loading images...")           
            self.input_im = load_image(self.image_path)
            self.input_im_all = [self.input_im]
            if self.load_gt_depth:
                self.depth_gt = load_depth(self.gt_depth_path)

    def load_latent(self):
        with torch.no_grad():
            def get_w_img(w_path):
                latent_w = torch.load(w_path, map_location='cpu')
                if type(latent_w) is dict:
                    latent_w = latent_w['latent']
                if latent_w.dim() == 1:
                    latent_w = latent_w.unsqueeze(0)
                latent_w = latent_w.cuda()

                return latent_w

            if self.joint_train:
                assert type(self.w_path) is list
                self.latent_w_all= []
                for w_path in self.w_path:
                    latent_w = get_w_img(w_path)
                    self.latent_w_all.append(latent_w.cpu())
                    
                self.latent_w = self.latent_w_all[0].cuda()
                
            else:
                if type(self.w_path) is list:
                    assert len(self.w_path) == self.world_size
                    self.w_path = self.w_path[self.rank]
                self.latent_w = get_w_img(self.w_path)
                self.latent_w_all= [self.latent_w]

    def setup_input(self, idx=None, epoch=None):
        if idx==None and epoch==None:
            self.image_path = self.img_list
            self.gt_depth_path = self.depth_list if self.load_gt_depth else None
            self.w_path = self.latent_list
        elif idx is not None:
            self.image_path=self.img_list[idx:idx+self.world_size]
            self.gt_depth_path=self.depth_list[idx:idx+self.world_size] if self.load_gt_depth else None
            self.w_path=self.latent_list[idx:idx+self.world_size]
        elif epoch is not None:
            self.image_path = self.img_list[epoch]
            self.gt_depth_path=self.depth_list[epoch] if self.load_gt_depth else None
            self.w_path=self.latent_list[epoch]
        else:
            print('setup_input param error')
        self.load_data()
        self.load_latent()

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix

    def set_epoch(self, epoch):
        self.epoch=epoch

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __repr__(self):
        """Print the number of instance number."""
        dataset_type = 'Test' if self.test_mode else 'Train'
        result = (f'\n{self.__class__.__name__} {dataset_type} dataset '
                  f'with number of images {len(self)} \n')
        if self.CLASSES is None:
            result += 'Category names are not provided. \n'
        else:
            result += f'Include category {self.CLASSES} \n'
        return result
        