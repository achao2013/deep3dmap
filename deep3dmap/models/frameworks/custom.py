# Copyright (c) achao2013. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import deep3dmap
import numpy as np
import torch
import torch.distributed as dist



class CustomFramework(metaclass=ABCMeta):
    """Base class for reconstructors."""

    def __init__(self, init_cfg=None):
        self.fp16_enabled = False
        self.loss_reduce = True

    @property
    def with_view(self):
        """bool: whether the reconstructor has a view head"""
        return hasattr(self, 'view_head') and self.view_head is not None

    @property
    def with_light(self):
        """bool: whether the reconstructor has a light head"""
        return (hasattr(self, 'light_head') and self.light_head is not None)

    @property
    def with_depth(self):
        """bool: whether the reconstructor has a depth head"""
        return (hasattr(self, 'depth_head') and self.depth_head is not None)

    @property
    def with_mask(self):
        """bool: whether the reconstructor has a mask head"""
        return (hasattr(self, 'mask_head') and self.mask_head is not None)

    @property
    def with_albedo(self):
        """bool: whether the reconstructor has a albedo head"""
        return (hasattr(self, 'albedo_head') and self.albedo_head is not None)

    @property
    def with_encoder(self):
        """bool: whether the reconstructor has a albedo head"""
        return (hasattr(self, 'encoder_head') and self.encoder_head is not None)

    def forward_train(self, inputs, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        pass


    def forward_test(self, inputs, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        pass

    def forward(self, inputs, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        pass
        """
        example

        if self.training:
            return self.forward_train(inputs, **kwargs)
        else:
            return self.forward_test(inputs, return_loss, **kwargs)
        """

