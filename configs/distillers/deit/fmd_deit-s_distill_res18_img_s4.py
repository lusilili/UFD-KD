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
    with_cls_token=True,
    teacher_pretrained = 'ckpt/deit-small-4xb256_in1k_20231207-vail01.pth',
    teacher_cfg = 'configs/deit/deit-small_pt-4xb256_in1k.py',
    student_cfg ='configs/resnet/resnet18_8xb32_in1k.py',
    distill_cfg = [dict(methods=[dict(type='FreqMaskingDistillLoss',
                                       name='loss_fmd_s4',
                                       use_this=jfd,
                                       alpha_jfd=[0.08, 0.000004],
                                       student_dims=512,
                                       teacher_dims=384,
                                       query_hw=(7,7),
                                       pos_hw=(7,7),
                                       pos_dims=384,
                                       self_query=True,
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
# deit-tiny_pt-4xb256_in1k_with_token

