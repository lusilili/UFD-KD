# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.05),
    clip_grad=dict(max_norm=5.0),
    # optimizer=dict(
    #     type='AdamW',
    #     lr=0.0005,
    #     weight_decay=0.05),
)

# learning policy
# param_scheduler = dict(
#     type='MultiStepLR', by_epoch=True, milestones=[100, 150], gamma=0.1)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=5e-7 / 5e-4,  # Warmup starting from 5e-7 to 5e-4
        by_epoch=True,
        begin=0,
        end=80  # Warmup for 20 epochs
    ),
    dict(
        type='CosineAnnealingLR',  # Using cosine annealing scheduler
        by_epoch=True,
        T_max=220,  # Total epochs - warmup_epochs = 300 - 20 = 280
        eta_min=1e-5  # Minimum learning rate
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=128)
