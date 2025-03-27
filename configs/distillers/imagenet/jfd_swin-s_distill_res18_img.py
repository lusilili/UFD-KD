_base_ = [
    '../../resnet/resnet18_8xb32_in1k.py'
]
# model settings
find_unused_parameters = True

# distillation settings
use_logit = False
is_vit = False

# config settings
vitkd = False
jfd = True
kd = False
# method details
model = dict(
    _delete_ = True,
    type='ClassificationDistiller',
    use_logit = use_logit,
    is_vit = is_vit,
    teacher_pretrained = 'ckpt/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth',
    teacher_cfg = 'configs/swin_transformer/swin-small_16xb64_in1k.py',
    student_cfg ='configs/resnet/resnet18_8xb32_in1k.py',
    distill_cfg = [dict(methods=[dict(type='JFDLoss',
                                       name='loss_jfd_s1',
                                       use_this=jfd,
                                       alpha_jfd=0.00,
                                       student_dims=64,
                                       teacher_dims=96,
                                       hw_shapes=(56,56),
                                       pos_dims=96,
                                       window_shapes=(4,4),
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='JFDLoss',
                                       name='loss_jfd_s2',
                                       use_this=jfd,
                                       alpha_jfd=0.00,
                                       student_dims=128,
                                       teacher_dims=192,
                                       hw_shapes=(28,28),
                                       pos_dims=192,
                                       window_shapes=(2,2),
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='JFDLoss',
                                       name='loss_jfd_s3',
                                       use_this=jfd,
                                       alpha_jfd=0.08,
                                       student_dims=256,
                                       teacher_dims=384,
                                       hw_shapes=(14,14),
                                       pos_dims=384,
                                       window_shapes=(1,1),
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='JFDLoss',
                                       name='loss_jfd_s4',
                                       use_this=jfd,
                                       alpha_jfd=0.08,
                                       student_dims=512,
                                       teacher_dims=768,
                                       hw_shapes=(7,7),
                                       pos_dims=768,
                                       window_shapes=(1,1),
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
