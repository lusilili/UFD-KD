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
    dict(crop_size=224, padding=4, type='RandomCrop'),
    dict(scale=(224, 224), type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    # dict(
    #     type='RandAugment',
    #     policies='rand-m9-mstd0.5-inc1',  # rand-m9-mstd0.5-inc1 policy
    #     num_policies=2,
    #     total_level=10,
    #     magnitude_level=9,
    #     magnitude_std=0.5,
    #     hparams=dict(interpolation='bicubic', pad_val=[104, 116, 124]),
    # ),
    dict(brightness=0.4, contrast=0.4, hue=0.1, saturation=0.4, type='ColorJitter'),  # Color jitter with 0.4
    # dict(type='Mixup', alpha=0.8, prob=1.0),  # Mixup with alpha=0.8
    # dict(type='CutMix', alpha=1.0, prob=1.0),  # CutMix with alpha=1.0
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(scale=(224, 224), type='Resize'),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=4,
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
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/cifar100/',
        pipeline=test_pipeline,
        test_mode=True
    ),
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
