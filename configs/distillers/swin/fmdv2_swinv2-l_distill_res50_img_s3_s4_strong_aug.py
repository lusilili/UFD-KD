_base_ = [
    '../../resnet/resnet50d_8xb32_in1k_strong_aug_coslr_300.py'
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
# /code/fanjiawei/fanjiawei/code/cls_KD_remote/configs/swin_transformer_v2/swinv2-base-w16_in21k-pre_16xb64_in1k-256px.py
# method details
model = dict(
    _delete_ = True,
    type='ClassificationDistiller',
    use_logit = use_logit,
    is_vit = is_vit,
    sync_bn=False,
    teacher_pretrained = 'ckpt/swinv2-large-w16_in21k-pre_3rdparty_in1k-256px_20220803-c40cbed7111.pth',
    teacher_cfg = 'configs/swin_transformer_v2/swinv2-large-w16_in21k-pre_16xb64_in1k-256px.py',
    student_cfg ='configs/resnet/resnet50d_8xb32_in1k_strong_aug_coslr_300.py',
    distill_cfg = [dict(methods=[dict(type='FreqMaskingDistillLossv2',
                                       name='loss_fmd_s3',
                                       use_this=jfd,
                                       alpha_jfd=[0.06, 0.045],
                                       student_dims=1024,
                                       teacher_dims=768,
                                       query_hw=(16,16),
                                       pos_hw=(16,16),
                                       window_shapes=(1,1),
                                       pos_dims=768,
                                       self_query=True,
                                       softmax_scale=[5.,5.],
                                       num_heads=16,
                                       dis_freq='high'
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='FreqMaskingDistillLossv2',
                                       name='loss_fmd_s4',
                                       use_this=jfd,
                                       alpha_jfd=[0.06, 0.045],
                                       student_dims=2048,
                                       teacher_dims=1536,
                                       query_hw=(8,8),
                                       pos_hw=(8,8),
                                       pos_dims=1536,
                                       self_query=False,
                                       softmax_scale=[5.,5.],
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

