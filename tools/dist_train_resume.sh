#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     tools/train.py \
#     $CONFIG \
#     --resume ckpt/eva02-base-p14_pre_in21k_20230505-2f2d4d3c.pth
#     --launcher pytorch

torchrun \
    --nnodes=1 \
    --nproc_per_node=$GPUS \
    --master_port=29500 \
    tools/train.py \
    $CONFIG \
    --launcher pytorch \
    --resume $RESUME  # 确保参数名正确

        # --dist-url tcp://${MLP_WORKER_0_HOST}:${MLP_WORKER_0_PORT} \
        # --num_machines ${MLP_WORKER_NUM} \
        # -d ${MLP_WORKER_GPU} \
        # --machine_rank ${MLP_ROLE_INDEX} \