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
    distill_cfg = [
     # dict(methods=[dict(type='JFDLossv2',
     #                                   name='loss_jfd_s3',
     #                                   use_this=jfd,
     #                                   alpha_jfd=0.1,
     #                                   student_dims=256,
     #                                   teacher_dims=384,
     #                                   hw_shapes=(14,14),
     #                                   window_shapes=(1,1),
     #                                   pos_dims=384,
     #                                   )
     #                              ]
     #                     ),
                    dict(methods=[dict(type='JFDLossv2',
                                       name='loss_jfd_s4',
                                       use_this=jfd,
                                       alpha_jfd=0.08,
                                       student_dims=512,
                                       teacher_dims=768,
                                       hw_shapes=(7,7),
                                       pos_dims=768,
                                       self_query=True
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
