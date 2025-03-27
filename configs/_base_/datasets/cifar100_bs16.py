# dataset settings
dataset_type = 'CIFAR100'
data_preprocessor = dict(
    num_classes=100,
    # RGB format normalization parameters
    mean=[129.304, 124.070, 112.434],
    std=[68.170, 65.392, 70.418],

    # loaded images are already RGB format
    to_rgb=True)

train_pipeline = [
    dict(type='RandomCrop', crop_size=64, padding=4),
    dict(scale=(64, 64), type='RandomResizedCrop'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(   brightness=0.4,
            contrast=0.4,
            hue=0.1,
            saturation=0.4,
            type='ColorJitter'),
    # dict(alpha=0.8, type='Mixup'),  # Mixup
    # dict(alpha=1.0, type='CutMix'),  # CutMix
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='Resize', scale=(64, 64)),
    # dict(
    #     type='ResizeEdge',
    #     scale=256,
    #     edge='short',
    #     backend='pillow',
    #     interpolation='bicubic'),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/cifar100',
        test_mode=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/cifar100/',
        test_mode=True,
        pipeline=test_pipeline),
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
