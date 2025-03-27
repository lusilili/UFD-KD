_base_ = [
    '../_base_/models/mobilenet_v2_1x_cifar.py',
    '../_base_/datasets/cifar100_mobilev2.py',
    '../_base_/schedules/cifar100_bs128_mobilev2.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        # out_indices=(2,3,4,7) 0-7
        out_indices=(1,3,5,7)
        )
    )