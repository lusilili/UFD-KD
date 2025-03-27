# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4,
        weight_decay=0.05),
    # constructor='LearningRateDecayOptimWrapperConstructor',
    # paramwise_cfg=dict(
    #     layer_decay_rate=0.75,  # layer-wise lr decay factor
    #     norm_decay_mult=0.,
    #     flat_decay_mult=0.,
    #     custom_keys={
    #         '.cls_token': dict(decay_mult=0.0),
    #         '.pos_embed': dict(decay_mult=0.0)
    #     })
    )

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=5e-7 / 5e-4,  # Warmup starting from 5e-7 to 5e-4
        by_epoch=True,
        begin=0,
        end=40  # Warmup for 20 epochs
    ),
    dict(
        type='CosineAnnealingLR',  # Using cosine annealing scheduler
        by_epoch=True,
        T_max=260,  # Total epochs - warmup_epochs = 300 - 20 = 280
        eta_min=1e-5  # Minimum learning rate
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: auto_scale_lr is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=128)