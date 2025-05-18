CUDA_VISIBLE_DEVICES=0,1,2 \
python -m torch.distributed.launch --nproc_per_node=3 eval.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --dataset ade20k \
    --data data/ade20k \
    --save-dir eva_dir/ade20k \
    --pretrained work_dir/ade20k/3g/kd_deeplabv3_resnet18_ade20k_best_model.pth


CUDA_VISIBLE_DEVICES=0,1,2 \
python eval.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --dataset ade20k \
    --data data/ade20k \
    --save-dir eva_dir/ade20k \
    --pretrained work_dir/ade20k/3g/kd_deeplabv3_resnet18_ade20k_best_model.pth


python eval.py \
  --model deeplabv3 \
  --backbone resnet18 \
  --dataset ade20k \
  --data data/ade20k \
  --save-dir eva_dir/ade20k \
  --pretrained work_dir/ade20k/3g/kd_deeplabv3_resnet18_ade20k_best_model.pth \
  --blend 
  