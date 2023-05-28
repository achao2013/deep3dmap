## Installation
sudo apt install libsparsehash-dev  # you can try to install sparsehash with conda if you don't have sudo privileges.

1.install enviroment.

if your gcc version in your system >=7.2
```sh
conda env create -f requirements/environment.yaml
conda activate deep3dmap2
```
else you need to install gcc first.
```sh
conda env create deep3dmap2
conda activate deep3dmap2
conda install -c moussi gcc_impl_linux-64
conda install -c moussi gcc_linux-64
conda install -c moussi gxx_impl_linux-64
conda install -c moussi gxx_linux-64
conda env update -f requirements/environment.yaml
```
**note**:pytorch is include in requirements/environment.yaml, you can take out it and install it by yourself, we recommend the version >=1.6, and you can choose other cuda version according to your machine.

2.install python packages.

pip install -r requirements/requirements.txt

<details>
  <summary>[FAQ on installation torchsparse]</summary>

- `AttributeError: module 'torchsparse_backend' has no attribute 'hash_forward'`
- Clone `torchsparse` to a local directory. If you have done that, recompile and install `torchsparse` after removing the `build` folder.

- No sudo privileges to install `libsparsehash-dev`
- Install `sparsehash` in conda (included in `environment.yaml`) and run `export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include` before installing `torchsparse`.

- if you want to rebuild torchsparse, please rm build folder and .so files

- For other problems, you can also refer to the [FAQ](https://github.com/mit-han-lab/torchsparse/blob/master/docs/FAQ.md) in `torchsparse`.
</details>

<details>
  <summary>[FAQ on import tinycudann]</summary>
- 'undefined symbol:_ZN2at4_ops19empty_memory_format4callEN3c108ArrayRefIlEENS2_8optionalINS2_10ScalarTypeEEENS5_INS2_6LayoutEEENS5_INS2_6DeviceEEENS5_IbEENS5_INS2_12MemoryFormatEEE'
- Install version not match between tinycudann and pytorch, uninstall and reinstall them.
</details>

3.install pnpmodules

first specify cuda path with the same version for pytorch 
```shell
export PATH="/usr/local/cuda-11.xxx/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.xxx/lib64:$LD_LIBRARY_PATH"
```

- stylegan2
```shell
cd pnpmodules/stylegan2/stylegan2-pytorch/op/
python setup.py install
```

- neural_renderer

```shell
cd pnpmodules/neural_renderer
python setup.py install
```


- "ImportError: libc10.so: cannot open shared object file: No such file or directory"
- libc10.so is based on torch, import torch first

4.install cuda build operators.

To use the cuda ops, you need to execute the build operation:
```shell
 export CUDA_HOME=$CUDA_PATH_Vxx.x
 export PATH=$CUDA_HOME/bin:$PATH
 export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
 export TORCH_CUDA_ARCH_LIST="3.5 3.7 5.0 5.2 6.0 6.1 7.0 7.5 8.0"
 WITH_OPS=1 python setup.py build_ext
```
