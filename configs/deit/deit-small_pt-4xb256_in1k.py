# In small and tiny arch, remove drop path and EMA hook comparing with the
# original config
_base_ = [
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='deit-small',
        with_cls_token=False, 
        output_cls_token=False,
        img_size=224,
        patch_size=16),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=384,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    # train_cfg=dict(augments=[
    #     dict(type='Mixup', alpha=0.8),
    #     dict(type='CutMix', alpha=1.0)
    # ]),
)


# EMA Hook
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL', evaluate_on_ema=True)]


# Checkpoint Hook
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=30))

# data settings
train_dataloader = dict(batch_size=32)

# schedule settings
optim_wrapper = dict(
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=5.0),
)
