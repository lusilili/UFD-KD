#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG \
    --launcher pytorch

        # --dist-url tcp://${MLP_WORKER_0_HOST}:${MLP_WORKER_0_PORT} \
        # --num_machines ${MLP_WORKER_NUM} \
        # -d ${MLP_WORKER_GPU} \
        # --machine_rank ${MLP_ROLE_INDEX} \