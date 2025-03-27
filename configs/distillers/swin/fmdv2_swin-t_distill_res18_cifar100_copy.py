_base_ = [
    '../../resnet/resnet18_8xb16_cifar100.py'
]
# model settings
find_unused_parameters = True

# distillation settings
use_logit = True
is_vit = False

# config settings
vitkd = False
jfd = True
kd = True
# method details
model = dict(
    _delete_ = True,
    type='ClassificationDistiller',
    use_logit = use_logit,
    is_vit = is_vit,
    sync_bn=True,
    teacher_pretrained = 'work_dirs/assets/swin-t-epoch_104.pth',
    teacher_cfg = 'configs/swin_transformer/swin-tiny_cifar100.py',
    student_cfg ='configs/resnet/resnet18_8xb16_cifar100.py',
    distill_cfg = [dict(methods=[dict(type='FreqMaskingDistillLossv2',
                                       name='loss_fmd_s3',
                                       use_this=True,
                                       alpha_jfd=[0.05, 0.045],
                                       student_dims=256,
                                       teacher_dims=384,
                                       query_hw=(14,14),
                                       pos_hw=(14,14),
                                       window_shapes=(1,1),
                                       pos_dims=384,
                                       self_query=True,
                                       softmax_scale=[5., 5.],
                                       num_heads=12,
                                       dis_freq='high'
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='FreqMaskingDistillLossv2',
                                       name='loss_fmd_s4',
                                       use_this=jfd,
                                       alpha_jfd=[0.05, 0.045],
                                       student_dims=512,
                                       teacher_dims=768,
                                       query_hw=(7,7),
                                       pos_hw=(7,7),
                                       window_shapes=(1,1),
                                       pos_dims=768,
                                       self_query=True,
                                       softmax_scale=[5., 5.],
                                       num_heads=24,
                                       dis_freq='high'
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='KDLoss',
                                       name='loss_kd',
                                       use_this = kd,
                                       temp=1.0,
                                       alpha=0.5,
                                       )
                                ]
                        ),                

                   ]
    )

# 