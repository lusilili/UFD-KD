# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer', arch='tiny', img_size=224, drop_path_rate=0.1,
        patch_size=4,  # 更小的 patch size 适应 CIFAR-100 小分辨率
        window_size=7,  # 更小的窗口尺寸
        pad_small_map=True  # 适应小尺寸输入
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=768,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.0)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]
    ),
)
