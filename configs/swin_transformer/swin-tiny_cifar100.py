_base_ = [
    '../_base_/models/swin_transformer/tiny_cifar.py',
    '../_base_/datasets/cifar100_swin.py',
    '../_base_/schedules/cifar100_swin.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        out_indices=(0,1,2,3)),
        pretrained="work_dirs/assets/swin-t_epoch_71.pth"
    )

train_dataloader = dict(batch_size=128)

# EMA hook
custom_hooks = [
    dict(momentum=0.0001, priority='ABOVE_NORMAL', type='EMAHook'),
]

# Checkpoint hook
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=30))

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))
