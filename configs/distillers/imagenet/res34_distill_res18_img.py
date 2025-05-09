_base_ = [
    '../../resnet/resnet18_8xb32_in1k.py'
]
# model settings
find_unused_parameters = False

# distillation settings
use_logit = True

# config settings
srrl = False
mgd = False
wsld = False
dkd = False
kd = False
# kd = True
nkd = True

# method details
model = dict(
    _delete_ = True,
    type='ClassificationDistiller',
    use_logit = use_logit,
    teacher_pretrained = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth',
    teacher_cfg = 'configs/resnet/resnet34_8xb32_in1k.py',
    student_cfg = 'configs/resnet/resnet18_8xb32_in1k.py',
    distill_cfg = [ dict(methods=[dict(type='SRRLLoss',
                                       name='loss_srrl',
                                       use_this = srrl,
                                       student_channels = 512,
                                       teacher_channels = 512,
                                       alpha=1.0,
                                       beta=1.0,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='MGDLoss',
                                       name='loss_mgd',
                                       use_this = mgd,
                                       student_channels = 512,
                                       teacher_channels = 512,
                                       alpha_mgd=0.00007,
                                       lambda_mgd=0.15,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='WSLDLoss',
                                       name='loss_wsld',
                                       use_this = wsld,
                                       temp=2.0,
                                       alpha=2.5,
                                       num_classes=1000,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='DKDLoss',
                                       name='loss_dkd',
                                       use_this = dkd,
                                       temp=1.0,
                                       alpha=1.0,
                                       beta=0.5,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='NKDLoss',
                                       name='loss_nkd',
                                       use_this = nkd,
                                       temp=1.0,
                                       gamma=1.5,
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
