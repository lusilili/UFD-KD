# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNetV1d',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        drop_path_rate=0.00,
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)),
    # train_cfg=dict(augments=[
    #     dict(type='Mixup', alpha=0.2),
    #     dict(type='CutMix', alpha=1.0)
    # ]),
)
