# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNetV1d',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        drop_path_rate=0.1,
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=2048,
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss=dict(
            label_smooth_val=0.1, mode='original', type='LabelSmoothLoss'),
    ),
    init_cfg=[
        dict(bias=0.0, layer='Linear', std=0.02, type='TruncNormal'),
        dict(bias=0.0, layer='LayerNorm', type='Constant', val=1.0),
    ],
    train_cfg=dict(augments=[
        dict(alpha=0.8, type='Mixup'),
        dict(alpha=1.0, type='CutMix'),
    ]),
    )
