# Copyright (c) achao2013. All rights reserved.
import torch
import torch.nn as nn
from ..builder import MODELS
from deep3dmap.models.frameworks import BaseFramework
from deep3dmap.datasets.pipelines.formating import to_tensor
from collections import OrderedDict
import re
from ..builder import MODELS, build_backbone
import cv2
import numpy as np
import torch.distributed as dist
from deep3dmap.core.utils.device_transfer import to_cuda
from deep3dmap.models.losses import MaskL1Loss, L1Loss

@MODELS.register_module()
class faceimg2uv(nn.Module):
    def __init__(self, model_cfgs, train_cfg=None, test_cfg=None, pretrained=None):
        super(faceimg2uv, self).__init__()
        self.model_cfgs = model_cfgs
        self.backbone=build_backbone(model_cfgs.backbone)
        self.uv_kpt_ind = np.loadtxt(model_cfgs.uv_kpt_ind_file).astype(np.int32)
        mask=cv2.imread(model_cfgs.weightmaskfile).astype(float)
        face=cv2.imread(model_cfgs.facemaskfile).astype(float)
        mask*=face
        mask=mask.transpose(2,0,1).astype(float)
        mask/=np.max(mask)
        self.mask=to_cuda(to_tensor(mask))
        print('mask shape:',self.mask.shape)
        self.criterion=MaskL1Loss(self.mask)
        self.criterion_lm=L1Loss()

    def init_weights(self):
        pass

    def forward(self, inputs, return_loss=False):
        #print('inputs in forward:', inputs)
        outputs=dict()
        outputs['uvpos'] = self.backbone(inputs['faceimg'])
        kpt_res = outputs['uvpos'][:,:,self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:]]
        outputs['kpt']=kpt_res
        if return_loss:
            loss=dict()
            #print(inputs['faceimg'].shape,outputs['uvpos'].shape, inputs['gt_uvimg'].shape)
            loss['loss_uv'] = self.criterion(outputs['uvpos'], inputs['gt_uvimg'])
            kpt_tgt = inputs['gt_uvimg'][:,:,self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:]]
            loss['loss_kpt'] = self.criterion_lm(kpt_res, kpt_tgt)
            return loss, outputs
        else:
            
            outputs['gt_kpt_proj2d']=inputs['gt_kpt_proj2d']
            outputs['tform_mat']=inputs['tform_mat']
            #print("tform_mat:",outputs['tform_mat'])
            for key in outputs:
                outputs[key]=list(outputs[key])
            return outputs

    def train_step(self, inputs, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.
        """
        losses, preds = self(inputs, return_loss=True)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(inputs['faceimg']))

        return outputs

    def val_step(self, inputs, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        preds = self(inputs)
        

        outputs = dict(
            preds=preds, tform_mat=inputs['tfrom_mat'], gt_kpt_proj2d=inputs['gt_kpt_proj2d'], num_samples=len(input['faceimg']))

        return outputs

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
