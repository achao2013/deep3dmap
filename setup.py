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


def parse_requirements(fname='requirements/runtime.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


#install_requires = parse_requirements()
#print("install_requires:",install_requires)

""" try:
    # OpenCV installed via conda.
    import cv2  # NOQA: F401
    major, minor, *rest = cv2.__version__.split('.')
    if int(major) < 3:
        raise RuntimeError(
            f'OpenCV >=3 is required but {cv2.__version__} is installed')
except ImportError:
    # If first not installed install second package
    CHOOSE_INSTALL_REQUIRES = [('opencv-python-headless>=3',
                                'opencv-python>=3')]
    for main, secondary in CHOOSE_INSTALL_REQUIRES:
        install_requires.append(choose_requirement(main, secondary)) """


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
                includes='deep3dmap/core/ops/csrc/*',
                show_detailed=True,
                is_pytorch_extension=True,
            )
            define_macros += [('WITH_CUDA', None)]
            define_macros += [('HIP_DIFF', None)]
            cuda_args = os.getenv('CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            op_files = glob.glob('./deep3dmap/core/ops/csrc/pytorch/hip/*')
            extension = CUDAExtension
            include_dirs.append(os.path.abspath('./deep3dmap/core/ops/csrc/common/hip'))
        elif torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
            print(f'Compiling {ext_name} with CUDA')
            define_macros += [('WITH_CUDA', None)]
            cuda_args = os.getenv('CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            op_files = glob.glob('./deep3dmap/core/ops/csrc/pytorch/*.cpp') + \
                glob.glob('./deep3dmap/core/ops/csrc/pytorch/cuda/*.cu')
            extension = CUDAExtension
            include_dirs.append(os.path.abspath('./deep3dmap/core/ops/csrc/common'))
            include_dirs.append(os.path.abspath('./deep3dmap/core/ops/csrc/common/cuda'))
        else:
            print(f'Compiling {ext_name} without CUDA')
            op_files = glob.glob('./deep3dmap/core/ops/csrc/pytorch/*.cpp')
            extension = CppExtension
            include_dirs.append(os.path.abspath('./deep3dmap/core/ops/csrc/common'))

        print(op_files)
        print(include_dirs)
        print(define_macros)
        print(extra_compile_args)
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
    cmdclass=cmd_class,
    zip_safe=False)
    #packages=find_packages(),
    #include_package_data=True,
    #install_requires=install_requires,
