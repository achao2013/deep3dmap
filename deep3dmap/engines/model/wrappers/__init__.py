# Copyright (c) achao2013. All rights reserved.
from engines.utils.dl_utils import TORCH_VERSION
from engines.utils.version_utils import digit_version
from .distributed import MMDistributedDataParallel
from .seperate_distributed import MMSeparateDistributedDataParallel
from .utils import is_model_wrapper

__all__ = [
    'MMDistributedDataParallel', 'is_model_wrapper',
    'MMSeparateDistributedDataParallel'
]

if digit_version(TORCH_VERSION) >= digit_version('1.11.0'):
    from .fully_sharded_distributed import \
        MMFullyShardedDataParallel  # noqa:F401
    __all__.append('MMFullyShardedDataParallel')
