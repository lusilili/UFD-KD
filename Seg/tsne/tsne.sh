CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 save_embeddings.py \
    --model deeplab_mobile \
    --backbone mobilenetv2 \
    --dataset citys \
    --data ../data/cityscapes/ \
    --save-dir ./ \
    --pretrained /data2/anbang/fjw/kd_codebase/CIRKD/vis_ckpt/student.pth

python tsne_visual.py




python tsne/save_embeddings.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --dataset ade20k \
    --data data/ade20k \
    --save-dir visual/resnet18 \
    --pretrained work_dir/ade20k/3g/kd_deeplabv3_resnet18_ade20k_best_model.pth \
    --crop-size 512 512

python tsne/tsne_visual.py



python tsne/save_embeddings.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --dataset ade20k \
    --data data/ade20k \
    --save-dir visual/resnet18 \
    --pretrained work_dir/ade20k/ori/kd_deeplabv3_resnet18_ade20k_test.pth \
    --crop-size 512 512

python tsne/tsne_visual.py