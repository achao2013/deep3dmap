## Installation
sudo apt install libsparsehash-dev  # you can try to install sparsehash with conda if you don't have sudo privileges.

conda env create -f requirements/environment.yaml

conda activate deep3dmap2

pip install -r requirements/requirements.txt

<details>
  <summary>[FAQ on installation]</summary>

- `AttributeError: module 'torchsparse_backend' has no attribute 'hash_forward'`
- Clone `torchsparse` to a local directory. If you have done that, recompile and install `torchsparse` after removing the `build` folder.

- No sudo privileges to install `libsparsehash-dev`
- Install `sparsehash` in conda (included in `environment.yaml`) and run `export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include` before installing `torchsparse`.

- For other problems, you can also refer to the [FAQ](https://github.com/mit-han-lab/torchsparse/blob/master/docs/FAQ.md) in `torchsparse`.
</details>

If you want to use the cuda ops, you need to execute the build operation:
```shell
 export CUDA_HOME=$CUDA_PATH_V10_2
 export PATH=$CUDA_HOME/bin:$PATH
 export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
 export TORCH_CUDA_ARCH_LIST="3.5 3.7 5.0 5.2 6.0 6.1 7.0 7.5"
 WITH_OPS=1 python setup.py build_ext
```
