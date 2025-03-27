_base_ = [
    '../../resnet/resnet50_8xb32_in1k.py'
]
# model settings
find_unused_parameters = True

# distillation settings
use_logit = True
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
    sync_bn=True,
    teacher_pretrained = 'ckpt/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth',
    teacher_cfg = 'configs/swin_transformer/swin-small_16xb64_in1k.py',
    student_cfg ='configs/resnet/resnet50_8xb32_in1k.py',
    distill_cfg = [dict(methods=[dict(type='FreqMaskingDistillLossv2',
                                       name='loss_fmd_s3',
                                       use_this=False,
                                       alpha_jfd=[0.08, 0.06],
                                       student_dims=1024,
                                       teacher_dims=384,
                                       query_hw=(14,14),
                                       pos_hw=(14,14),
                                       window_shapes=(1,1),
                                       pos_dims=384,
                                       self_query=True,
                                       softmax_scale=[5.,5.],
                                       num_heads=3,
                                       dis_freq='high'
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='FreqMaskingDistillLossv2',
                                       name='loss_fmd_s4',
                                       use_this=jfd,
                                       alpha_jfd=[0.08, 0.00],
                                       student_dims=2048,
                                       teacher_dims=768,
                                       query_hw=(7,7),
                                       pos_hw=(7,7),
                                       pos_dims=768,
                                       self_query=True,
                                       softmax_scale=[5.,5.],
                                       num_heads=6,
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
