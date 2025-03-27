_base_ = [
    '../../deit/deit-tiny_pt-4xb256_cifar100.py'
]
# model settings
find_unused_parameters = True

# distillation settings
use_logit = True
is_vit = True

# config settings
srrl = False
mgd = False
wsld = False
dkd = False
kd = True
nkd = False
# method details
model = dict(
    _delete_ = True,
    type='ClassificationDistiller',
    use_logit = use_logit,
    teacher_pretrained = 'work_dirs/assets/mixer-b16-87.43.pth',
    teacher_cfg ='configs/mlp_mixer/mlp-mixer-base-p16_64xb64_cifar.py',
    student_cfg ='configs/deit/deit-tiny_pt-4xb256_cifar100.py',
    distill_cfg = [
                   dict(methods=[dict(type='UnifiedFreqDecoupleLoss',  
                            name='loss_ufdl',
                            use_this=True,
                            alpha_jfd=[0.03, 0.065, 0.003, 0.0065],  # 权重比例
                            student_dims=192,
                            teacher_dims=768,
                            query_hw=(14,14),
                            pos_hw=(14,14),
                            window_shapes=(1, 1),
                            pos_dims=768,
                            self_query=True,
                            softmax_scale=[5., 5.],
                            dis_freq='high',
                            num_heads=32)
                        ]),
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

