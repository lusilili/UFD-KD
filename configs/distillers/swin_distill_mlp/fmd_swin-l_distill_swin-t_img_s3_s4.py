_base_ = [
    '../../swin_transformer/swin-tiny_16xb64_in1k.py'
]
# model settings
find_unused_parameters = True

# distillation settings
use_logit = True
is_vit = False

# configs/swin_transformer/swin-tiny_16xb64_in1k.py
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
    sync_bn=False,
    teacher_pretrained = 'ckpt/swin_large_patch4_window7_224_22kto1k-5f0996db.pth',
    teacher_cfg = 'configs/swin_transformer/swin-large_16xb64_in1k.py',
    student_cfg ='configs/swin_transformer/swin-tiny_16xb64_in1k.py',
    distill_cfg = [dict(methods=[dict(type='FreqMaskingDistillLossConvNext',
                                       name='loss_fmd_s3',
                                       use_this=True,
                                       alpha_jfd=[0.08, 0.06],
                                       student_dims=384,
                                       teacher_dims=768,
                                       query_hw=(14,14),
                                       pos_hw=(14,14),
                                       pos_dims=768,
                                       self_query=True,
                                       num_heads=24,
                                       softmax_scale=[1.0, 1.0],
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='FreqMaskingDistillLossConvNext',
                                       name='loss_fmd_s4',
                                       use_this=True,
                                       alpha_jfd=[0.08, 0.06],
                                       student_dims=768,
                                       teacher_dims=1536,
                                       query_hw=(7,7),
                                       pos_hw=(7,7),
                                       pos_dims=1536,
                                       self_query=False,
                                       num_heads=48,
                                       softmax_scale=[1.0, 1.0],
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

