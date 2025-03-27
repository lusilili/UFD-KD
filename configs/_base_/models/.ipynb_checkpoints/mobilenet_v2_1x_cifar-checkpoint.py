# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=1280,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0),
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
    train_cfg=dict(device_ids=[0, 1])  # 根据实际 GPU ID 调整
    )
