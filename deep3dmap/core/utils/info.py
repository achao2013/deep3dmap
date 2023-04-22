# Copyright (c) achao2013. All rights reserved.
import glob
import os

import torch


from ..utils import ext_loader
#ext_module = ext_loader.load_ext(
#    '_ext', ['get_compiler_version', 'get_compiling_cuda_version'])

def get_compiler_version():
    return '0.0.0'
    #return ext_module.get_compiler_version()

def get_compiling_cuda_version():
    return '10.2'
    #return ext_module.get_compiling_cuda_version()


