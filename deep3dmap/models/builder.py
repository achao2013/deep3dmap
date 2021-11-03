#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: achao
# File Name: builder.py
# Description:
"""

import warnings


from deep3dmap.core.utils import Registry


from deep3dmap.runners import Sequential
from deep3dmap.core.utils import Registry, build_from_cfg


def build_model_from_cfg(cfg, registry, default_args=None):
    """Build a PyTorch model from config dict(s). Different from
    ``build_from_cfg``, if cfg is a list, a ``nn.Sequential`` will be built.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a config
            dict or a list of config dicts. If cfg is a list, a
            the built modules will be wrapped with ``nn.Sequential``.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


MODELS = Registry('model', build_func=build_model_from_cfg)



BACKBONES = MODELS

LOSSES = MODELS

RECONSTRUCTORS = MODELS

def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)





def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_reconstruction(cfg, train_cfg=None, test_cfg=None):
    """Build reconstruct."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return RECONSTRUCTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
