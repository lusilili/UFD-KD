#!/usr/bin/env bash
CONFIG='configs/resnet/jfd_swin-s_distill_res18_img.py'

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/train.py \
    --dist-url tcp://${MLP_WORKER_0_HOST}:${MLP_WORKER_0_PORT}\
    --num_machines ${MLP_WORKER_NUM}\
    -d ${MLP_WORKER_GPU}\
    --machine_rank ${MLP_ROLE_INDEX}\
    $CONFIG \
    --launcher pytorch ${@:3}

        # --dist-url tcp://${MLP_WORKER_0_HOST}:${MLP_WORKER_0_PORT} \
        # --num_machines ${MLP_WORKER_NUM} \
        # -d ${MLP_WORKER_GPU} \
        # --machine_rank ${MLP_ROLE_INDEX} \