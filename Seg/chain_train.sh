#!/bin/bash
TARGET_CMD="train_ufd.py --teacher-model deeplabv3 --student-model deeplabv3 --log-dir log_dir/ade20k/"
PIDS=$(pgrep -f "$TARGET_CMD")

[ -z "$PIDS" ] && echo "未找到进程" && exit 1

echo "监控进程列表："
ps -f -p $PIDS

while : 
do
    RUNNING=$(ps -o pid= -p $PIDS | wc -l)
    [ $RUNNING -eq 0 ] && break

    echo "[$(date +"%F %T")] 剩余进程数: $RUNNING"

    UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk "{sum+=\$1} END {print sum/NR}")
    [ $UTIL -lt 5 ] && echo "GPU利用率低，终止进程" && kill $PIDS

    sleep 300
done

MAX_RETRY=3
for ((i=1; i<=$MAX_RETRY; i++)); do
    echo "启动新训练（第$i次）..."
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_ufd.py \
        --teacher-model deeplabv3 \
        --student-model deeplabv3 \
        --teacher-backbone resnet101 \
        --student-backbone resnet18 \
        --lambda-kd 0.5 \
        --dataset ade20k \
        --data data/ade20k \
        --batch-size 16 \
        --workers 2 \
        --crop-size 512 512 \
        --lr 0.015 \
        --max-iterations 40000 \
        --save-dir work_dir/ade20k/ \
        --log-dir log_dir/ade20k/lr-0.015 \
        --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_ade20k_best_model.pth \
        --student-pretrained-base pretrained_ckpt/resnet18-imagenet.pth

    [ $? -eq 0 ] && break || sleep 60
done
