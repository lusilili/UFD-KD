_base_ = [
    '../_base_/models/mlp_mixer_base_patch16_cifar.py',
    '../_base_/datasets/cifar100_mixer.py',
    '../_base_/schedules/cifar100_bs128_mixer.py',
    '../_base_/default_runtime.py',
]

# configs/_base_/models/mlp_mixer_base_patch14.py

model = dict(
        pretrained="work_dirs/assets/mixer-b-16_e300_scalekd_from_swin-l.pth"
    )

optim_wrapper = dict(clip_grad=dict(max_norm=1.0))
