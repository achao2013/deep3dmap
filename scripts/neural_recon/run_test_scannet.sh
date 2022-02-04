#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
	PYOPENGL_PLATFORM=osmesa python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
	    $(dirname "$0")/../../tools/test.py $CONFIG $CHECKPOINT --launcher=pytorch --eval=depth_mesh
