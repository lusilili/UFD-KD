_base_ = [
    '../_base_/models/convnext/convnext-tiny-cifar.py',
    '../_base_/datasets/cifar100_convnext.py',
    '../_base_/schedules/cifar100_bs128_convnext.py',
    '../_base_/default_runtime.py',
]

model = dict(
        pretrained="work_dirs/assets/convnext-t_e300_scalekd_from_swin-l.pth"
    )


# schedule setting
# optim_wrapper = dict(
#     optimizer=dict(lr=4e-3),
#     # clip_grad=None,
#     clip_grad=dict(max_norm=5.0)
# )

# # grad clip
# optim_wrapper = dict(clip_grad=dict(max_norm=1.0))

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=30))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)
