#!/usr/bin/env bash
#EXP=celeba
#CONFIG=celeba
#GPUS=4
#PORT=${PORT:-29579}

#mkdir -p results/${EXP}
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    run.py \
#    --launcher pytorch \
#    --config configs/${CONFIG}.yml \
#    2>&1 | tee results/${EXP}/log.txt


CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
	python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
	$(dirname "$0")/../../tools/train.py $CONFIG --no-validate --launcher pytorch
