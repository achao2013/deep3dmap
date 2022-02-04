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
import os.path as osp

@DATASETS.register_module()
class AFLW2000Dataset(CustomDataset):
    CLASSES = ('face')
    def __init__(self, datapath, img_prefix='',pipeline=None,test_mode=False):
        self.img_prefix = img_prefix
        lines = open(datapath).readlines()
        self.data_infos=[]
        for line in lines:
            info={'filename':line.strip()}
            if osp.exists(osp.join(img_prefix,info['filename'])) and \
                 osp.exists(osp.join(img_prefix,info['filename']).replace('.jpg','.mat')):
                self.data_infos.append(info)
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.flag = np.zeros(len(self), dtype=np.uint8)
        
    def __len__(self):
        return len(self.data_infos)
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['mat_prefix'] = self.img_prefix
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        mat_info = {'filename':img_info['filename'].replace('.jpg','.mat')}
        results = dict(img_info=img_info, mat_info=mat_info)
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
        mat_info = {'filename':img_info['filename'].replace('.jpg','.mat')}
        results = dict(img_info=img_info, mat_info=mat_info)
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

    def evaluate(self, results, metric='nme', logger=None, scale_ranges=None):
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['nme', 'rmse']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        eval_results = OrderedDict()
        if metric=='nme':
            nmelist=[]
            print('in evaluate kpt size:',len(results['kpt']))
            
            for key in results:
                results[key]=np.stack([r.cpu().numpy() for r in results[key]])
            kpt_res = results['kpt']
            kpt_res68 = kpt_res[:,:2,:]*255.0
            origin_kpt2d=[]
            for j in range(kpt_res68.shape[0]):
                cropped_vertices = np.vstack((kpt_res68[j], np.ones((1,68))))
                origin_cord = np.dot(np.linalg.inv(results['tform_mat'][j]), cropped_vertices)
                origin_kpt2d.append(origin_cord[:2,:].transpose(1,0))
            origin_kpt2d=np.array(origin_kpt2d)
            gt_kpt_proj2d = results['gt_kpt_proj2d'].transpose(0,2,1)
            
            num = origin_kpt2d.shape[0]
            w = np.array([abs(np.max(gt_kpt_proj2d[i,:,0])-np.min(gt_kpt_proj2d[i,:,0])) for i in range(num)])
            h = np.array([abs(np.max(gt_kpt_proj2d[i,:,1])-np.min(gt_kpt_proj2d[i,:,1])) for i in range(num)])
            #print w,h
            nmelist.extend(list(np.squeeze(np.mean(np.squeeze(np.sqrt(np.sum(np.square(gt_kpt_proj2d-origin_kpt2d),2))),1))/np.sqrt(w*h)))
            eval_results['nme']=sum(nmelist)/len(nmelist)
        return eval_results
        