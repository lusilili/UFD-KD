_base_ = [
    '../../mobilenet_v1/mobilenet_v1.py'
]
# model settings
find_unused_parameters = True

# distillation settings
use_logit = True
is_vit = True

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
    teacher_pretrained = 'ckpt/deit-small-4xb256_in1k_20231207-vail01.pth',
    teacher_cfg = 'configs/deit/deit-small_pt-4xb256_in1k.py',
    student_cfg ='configs/mobilenet_v1/mobilenet_v1.py',
    distill_cfg = [dict(methods=[dict(type='FreqMaskingDistillLoss',
                                       name='loss_fmd_s3',
                                       use_this=jfd,
                                       alpha_jfd=[0.02, 0.008],
                                       student_dims=512,
                                       teacher_dims=384,
                                       query_hw=(14,14),
                                       pos_hw=(14,14),
                                       pos_dims=384,
                                       softmax_scale=[5, 5],
                                       num_heads=6,
                                       self_query=True,
                                       dis_freq='high'
                                       )
                                  ]
                         ),
        dict(methods=[dict(type='FreqMaskingDistillLoss',
                                       name='loss_fmd_s4',
                                       use_this=jfd,
                                       alpha_jfd=[0.02, 0.008],
                                       student_dims=1024,
                                       teacher_dims=384,
                                       query_hw=(14,14),
                                       pos_hw=(7,7),
                                       pos_dims=384,
                                       softmax_scale=[5, 5],
                                       num_heads=6,
                                       self_query=False,
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