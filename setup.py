import glob
import os
import re
from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup

EXT_TYPE = ''
try:
    import torch
    if True:
        from torch.utils.cpp_extension import BuildExtension
        EXT_TYPE = 'pytorch'
    cmd_class = {'build_ext': BuildExtension}
except ModuleNotFoundError:
    cmd_class = {}
    print('Skip building ext ops due to the absence of torch.')

def get_extensions():
    extensions = []
    if True:
        ext_name = 'deepmap3d._ext'
        from torch.utils.cpp_extension import CppExtension, CUDAExtension

        # prevent ninja from using too many resources
        try:
            import psutil
            num_cpu = len(psutil.Process().cpu_affinity())
            cpu_use = max(4, num_cpu - 1)
        except (ModuleNotFoundError, AttributeError):
            cpu_use = 4

        os.environ.setdefault('MAX_JOBS', str(cpu_use))
        define_macros = []
        extra_compile_args = {'cxx': []}
        include_dirs = []

        is_rocm_pytorch = False
        try:
            from torch.utils.cpp_extension import ROCM_HOME
            is_rocm_pytorch = True if ((torch.version.hip is not None) and
                                       (ROCM_HOME is not None)) else False
        except ImportError:
            pass

        project_dir = 'mmcv/ops/csrc/'
        if is_rocm_pytorch:
            from torch.utils.hipify import hipify_python

            hipify_python.hipify(
                project_directory=project_dir,
                output_directory=project_dir,
                includes='mmcv/ops/csrc/*',
                show_detailed=True,
                is_pytorch_extension=True,
            )
            define_macros += [('MMCV_WITH_CUDA', None)]
            define_macros += [('HIP_DIFF', None)]
            cuda_args = os.getenv('MMCV_CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/hip/*')
            extension = CUDAExtension
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common/hip'))
        elif torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
            define_macros += [('MMCV_WITH_CUDA', None)]
            cuda_args = os.getenv('MMCV_CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cu')
            extension = CUDAExtension
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common/cuda'))
        else:
            print(f'Compiling {ext_name} without CUDA')
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp')
            extension = CppExtension
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))

        ext_ops = extension(
            name=ext_name,
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args)
        extensions.append(ext_ops)

setup(
    name='deep3dmap' if os.getenv('WITH_OPS', '0') == '0' else 'deep3dmap-full',
    ext_modules=get_extensions(),
    cmdclass=cmd_class)