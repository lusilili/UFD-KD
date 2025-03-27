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
    distill_cfg = [dict(methods=[dict(type='FDLoss',
                                       name='loss_fd',
                                       use_this=jfd,
                                       alpha_fd=0.08,
                                       student_dims=2048,
                                       teacher_dims=768,
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='KDLoss',
                                       name='loss_kd',
                                       use_this=False,
                                       temp=1.0,
                                       alpha=0.5,
                                       )
                                ]
                        ),                

                   ]
    )
