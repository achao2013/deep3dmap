# Copyright (c) achao2013. All rights reserved.
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
class ThreeHundredWLPDataset(CustomDataset):
    CLASSES = ('face')
    def __init__(self, datapath, img_prefix='',pipeline=None,test_mode=False):
        self.img_prefix = img_prefix
        lines = open(datapath).readlines()
        self.data_infos=[]
        for line in lines:
            info={'filename':line.strip().replace('.jpg','_inp.jpg')}
            #print(osp.exists(osp.join(img_prefix,info['filename'])), osp.exists(osp.join(img_prefix,info['filename']).replace('.jpg','.npy')))
            if osp.exists(osp.join(img_prefix,info['filename'])) and \
                 osp.exists(osp.join(img_prefix,info['filename']).replace('_inp.jpg','.npy')):
                self.data_infos.append(info)
        print('dataset len:',len(self.data_infos))
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __len__(self):
        return len(self.data_infos)
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['arr_prefix'] = self.img_prefix
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        arr_info = {'filename':img_info['filename'].replace('_inp.jpg','.npy')}
        results = dict(img_info=img_info, arr_info=arr_info)
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

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        
        return np.random.choice(np.arange(len(self)))

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
        