# In small and tiny arch, remove drop path and EMA hook comparing with the
# original config
_base_ = [
    '../_base_/datasets/cifar100_resnet.py',
    '../_base_/schedules/cifar10_bs128_vit.py',
    '../_base_/default_runtime.py'
]

# paramwise_cfg = dict(custom_keys={
#     '.cls_token': dict(decay_mult=0.0),
#     '.pos_embed': dict(decay_mult=0.0)
# })

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        # arch='s',
        arch=dict(
            embed_dims=384,  # 注意这里是复数形式
            num_layers=12,    # 标准small的层数
            num_heads=6,      # 384/64=6
            feedforward_channels=1536  # 384 * 4
        ),
        with_cls_token=False, 
        output_cls_token=False,
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        drop_path_rate=0.1,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=384,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    pretrained="work_dirs/assets/vit-s-16_e300_scalekd_from_swin-l.pth",
    init_cfg=[
        dict(bias=0.0, layer='Linear', std=0.02, type='TruncNormal'),
        dict(bias=0.0, layer='LayerNorm', type='Constant', val=1.0),
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)

# EMA Hook
custom_hooks = [dict(type='EMAHook', momentum=0.0001, priority='ABOVE_NORMAL', evaluate_on_ema=True)]

# Checkpoint Hook
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=200))

# # data settings
# train_dataloader = dict(batch_size=32)

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
