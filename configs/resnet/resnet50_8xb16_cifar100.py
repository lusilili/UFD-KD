_base_ = [
    '../_base_/models/resnet50_cifar.py',
    '../_base_/datasets/cifar100_resnet.py',
    '../_base_/schedules/cifar10_bs128.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(head=dict(num_classes=100),
             pretrained="work_dirs/assets/resnet50_epoch600.pth")

# schedule settings
optim_wrapper = dict(optimizer=dict(weight_decay=0.05))

# EMA hook
custom_hooks = [
    dict(momentum=0.0001, priority='ABOVE_NORMAL', type='EMAHook'),
]

# Checkpoint hook
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))

# param_scheduler = dict(
#     type='MultiStepLR',
#     by_epoch=True,
#     milestones=[60, 120, 160],
#     gamma=0.2,
# )
