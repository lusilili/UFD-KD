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
    teacher_pretrained = 'work_dirs/assets/convnext-t-88.47.pth',
    teacher_cfg = 'configs/convnext/convnext-tiny_32xb128_cifar.py',
    student_cfg = 'configs/deit/deit-tiny_pt-4xb256_cifar100.py',
    distill_cfg = [ 
                   dict(methods=[dict(type='UnifiedFreqDecoupleLoss',  
                            name='loss_ufdl',
                            use_this=True,
                            alpha_jfd=[0.03, 0.065, 0.003, 0.0065],  # 权重比例
                            student_dims=192,
                            teacher_dims=768,
                            query_hw=(7,7),
                            pos_hw=(14,14),
                            window_shapes=(1, 1),
                            pos_dims=768,
                            self_query=True,
                            softmax_scale=[5., 5.],
                            dis_freq='high',
                            num_heads=32)
                        ]),
                #   dict(methods=[dict(type='SRRLLoss',
                #                        name='loss_srrl',
                #                        use_this = srrl,
                #                        student_channels = 512,
                #                        teacher_channels = 2048,
                #                        alpha=1.0,
                #                        beta=5.0,
                #                        )
                #                 ]
                #         ),
                #     dict(methods=[dict(type='MGDLoss',
                #                        name='loss_mgd',
                #                        use_this = mgd,
                #                        student_channels = 512,
                #                        teacher_channels = 2048,
                #                        alpha_mgd=0.0012,
                #                        lambda_mgd=0.15,
                #                        )
                #                 ]
                #         ),
                #     dict(methods=[dict(type='WSLDLoss',
                #                        name='loss_wsld',
                #                        use_this = wsld,
                #                        temp=4.0,
                #                        alpha=2.25,
                #                        num_classes=100,
                #                        )
                #                 ]
                #         ),
                #     dict(methods=[dict(type='DKDLoss',
                #                        name='loss_dkd',
                #                        use_this = dkd,
                #                        temp=4.0,
                #                        alpha=1.0,
                #                        beta=8.0,
                #                        )
                #                 ]
                #         ),
                #     dict(methods=[dict(type='NKDLoss',
                #                        name='loss_nkd',
                #                        use_this = nkd,
                #                        temp=3.0,
                #                        gamma=3.5,
                #                        )
                #                 ]
                #         ),
                    dict(methods=[dict(type='KDLoss',
                                       name='loss_kd',
                                       use_this = True,
                                       temp=1.0,
                                       alpha=0.5,
                                       )
                                ]
                        ),

                   ]
    )