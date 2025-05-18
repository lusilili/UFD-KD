CUDA_VISIBLE_DEVICES=0,1,2 \
python -m torch.distributed.launch --nproc_per_node=3 \
    train_ocmgd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --lambda-kd 0.5 \
    --lambda-fd 0.000000 \
    --lambda-ctr 0.0 \
    --dataset ade20k \
    --data data/ade20k \
    --batch-size 16 \
    --workers 8 \
    --crop-size 512 512 \
    --lr 0.02 \
    --max-iterations 40000 \
    --save-dir work_dir/ade20k/ori \
    --log-dir log_dir/ade20k/ori \
    --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_ade20k_best_model.pth  \
    --student-pretrained-base pretrained_ckpt/resnet18-imagenet.pth


CUDA_VISIBLE_DEVICES=0,1,2 \
python -m torch.distributed.launch --nproc_per_node=3 \
    train_ufd.py \
    --teacher-model deeplabv3 \
    --student-model deeplab_mobile \
    --teacher-backbone resnet101 \
    --student-backbone mobilenetv2 \
    --lambda-kd 0.5 \
    --dataset ade20k \
    --data data/ade20k \
    --batch-size 39 \
    --workers 3 \
    --crop-size 512 512 \
    --lr 0.02 \
    --max-iterations 40000 \
    --save-dir work_dir/ade20k/mbv/ufd \
    --log-dir log_dir/ade20k/mbv/ufd \
    --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_ade20k_best_model.pth  \
    --student-pretrained-base pretrained_ckpt/mobilenetv2-imagenet.pth
    

# kd
CUDA_VISIBLE_DEVICES=0,1,2 \
python -m torch.distributed.launch --nproc_per_node=3 \
    train_kd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --lambda-kd 0.5 \
    --dataset ade20k \
    --data data/ade20k \
    --batch-size 12 \
    --workers 3 \
    --crop-size 512 512 \
    --lr 0.02 \
    --max-iterations 40000 \
    --save-dir work_dir/ade20k/kd \
    --log-dir log_dir/ade20k/kd \
    --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_ade20k_best_model.pth  \
    --student-pretrained-base pretrained_ckpt/resnet18-imagenet.pth

    
    
CUDA_VISIBLE_DEVICES=0 python train_ufd.py --teacher-model deeplabv3 --student-model deeplabv3 --teacher-backbone resnet101 --student-backbone resnet18 --lambda-kd 0.5 --lambda-fd 0.000001 --dataset ade20k --data data/ade20k --batch-size 8 --workers 8 --crop-size 512 512 --lr 0.0001 --max-iterations 40000 --save-dir work_dir/ade20k/ --log-dir log_dir/ade20k/ --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_ade20k_best_model.pth --student-pretrained-base pretrained_ckpt/resnet18-imagenet.pth

# origin-ocmgd
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 train_ocmgd.py --teacher-model deeplabv3 --student-model deeplabv3 --teacher-backbone resnet101 --student-backbone resnet18 --lambda-kd 0.5 --lambda-fd 0.000001 --lambda-ctr 0.04 --dataset ade20k --data data/ade20k --batch-size 16 --workers 3 --crop-size 512 512 --lr 0.02 --max-iterations 40000 --save-dir work_dir/ade20k/ --log-dir log_dir/ade20k/ --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_ade20k_best_model.pth --student-pretrained-base pretrained_ckpt/resnet18-imagenet.pth


CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 train_ufd.py --teacher-model deeplabv3 --student-model deeplabv3 --teacher-backbone resnet101 --student-backbone resnet18 --lambda-kd 0.5 --lambda-fd 0.00004 --dataset ade20k --data data/ade20k --batch-size 40 --workers 3 --crop-size 512 512 --lr 0.0213 --max-iterations 40000 --save-dir work_dir/ade20k/ --log-dir log_dir/ade20k/ --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_ade20k_best_model.pth --student-pretrained-base pretrained_ckpt/resnet18-imagenet.pth


# compare-ocmgd
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_ocmgd.py --teacher-model deeplabv3 --student-model deeplabv3 --teacher-backbone resnet101 --student-backbone resnet18 --lambda-kd 0.5 --lambda-fd 0.000001 --lambda-ctr 0.04 --dataset ade20k --data data/ade20k --batch-size 16 --workers 2 --crop-size 512 512 --lr 0.02 --max-iterations 40000 --save-dir work_dir/ade20k/ --log-dir log_dir/ade20k/origin --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_ade20k_best_model.pth --student-pretrained-base pretrained_ckpt/resnet18-imagenet.pth



# opt 2
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 train_ufd.py --teacher-model deeplabv3 --student-model deeplabv3 --teacher-backbone resnet101 --student-backbone resnet18 --lambda-kd 0.5 --dataset ade20k --data data/ade20k --batch-size 24 --workers 3 --crop-size 512 512 --lr 0.02 --max-iterations 40000 --save-dir work_dir/ade20k/3g --log-dir log_dir/ade20k/3g --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_ade20k_best_model.pth --student-pretrained-base pretrained_ckpt/resnet18-imagenet.pth

# mbv
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 train_ufd.py --teacher-model deeplabv3 --student-model deeplabv3 --teacher-backbone resnet101 --student-backbone resnet18 --lambda-kd 0.5 --dataset ade20k --data data/ade20k --batch-size 33 --workers 2 --crop-size 512 512 --lr 0.02 --max-iterations 40000 --save-dir work_dir/ade20k/mbv --log-dir log_dir/ade20k/mbv --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_ade20k_best_model.pth --student-backbone mobilenetv2

CUDA_VISIBLE_DEVICES=2 python train_ufd.py --teacher-model deeplabv3 --student-model deeplabv3 --teacher-backbone resnet101 --student-backbone resnet18 --lambda-kd 0.5 --lambda-fd 0.000001 --dataset ade20k --data data/ade20k --batch-size 15 --workers 1 --crop-size 512 512 --lr 0.015 --max-iterations 40000 --save-dir work_dir/ade20k/ --log-dir log_dir/ade20k/1g/ --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_ade20k_best_model.pth --student-pretrained-base pretrained_ckpt/resnet18-imagenet.pth



CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 train_ufd.py --teacher-model deeplabv3 --student-model deeplabv3 --teacher-backbone resnet101 --student-backbone resnet18 --lambda-kd 0.5 --lambda-fd 0.000001 --dataset ade20k --data data/ade20k --batch-size 32 --workers 3 --crop-size 512 512 --lr 0.015 --max-iterations 40000 --save-dir work_dir/ade20k/ --log-dir log_dir/ade20k/ --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_ade20k_best_model.pth --student-pretrained-base pretrained_ckpt/resnet18-imagenet.pth
