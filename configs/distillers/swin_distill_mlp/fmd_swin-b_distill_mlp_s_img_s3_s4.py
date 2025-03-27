_base_ = [
    '../../mlp_mixer/mlp-mixer-small-p16_64xb64_in1k.py'
]
# model settings
find_unused_parameters = True

# distillation settings
use_logit = False
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
    teacher_pretrained = 'ckpt/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth',
    teacher_cfg = 'configs/swin_transformer/swin-base_16xb64_in1k.py',
    student_cfg ='configs/mlp_mixer/mlp-mixer-small-p16_64xb64_in1k.py',
    distill_cfg = [dict(methods=[dict(type='FreqMaskingDistillLossv2',
                                       name='loss_fmd_s3',
                                       use_this=jfd,
                                       alpha_jfd=[0.08, 0.06],
                                       student_dims=512,
                                       teacher_dims=512,
                                       query_hw=(14,14),
                                       pos_hw=(14,14),
                                       pos_dims=512,
                                       self_query=True,
                                       softmax_scale=5.0,
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='FreqMaskingDistillLossv2',
                                       name='loss_fmd_s4',
                                       use_this=jfd,
                                       alpha_jfd=[0.08, 0.06],
                                       student_dims=512,
                                       teacher_dims=1024,
                                       query_hw=(7,7),
                                       pos_hw=(14,14),
                                       pos_dims=1024,
                                       self_query=False,
                                       softmax_scale=5.0,
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


