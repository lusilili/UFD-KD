# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MlpMixer',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
        out_indices=[-7,-5,-3,-1],
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)
# model = dict(
#     type='ImageClassifier',
#     backbone=dict(
#         type='MlpMixer',
#         arch='b',
#         img_size=224,
#         patch_size=16,
#         drop_path_rate=0.1,
#         init_cfg=[
#             dict(
#                 type='Kaiming',
#                 layer='Conv2d',
#                 mode='fan_in',
#                 nonlinearity='linear')
#         ]),
#     neck=dict(type='GlobalAveragePooling', dim=1),
#     head=dict(
#         type='LinearClsHead',
#         num_classes=1000,
#         in_channels=768,
#         loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
#         topk=(1, 5),
#     ),
# )
