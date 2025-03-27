_base_ = [
    '../mobilenet_v2/mobilenet-v2_cifar100.py'
]
# model settings
find_unused_parameters = True

# distillation settings
use_logit = True
is_vit = False

# config settings
vitkd = True
jfd = True
kd = True
# method details
model = dict(
    _delete_ = True,
    type='ClassificationDistiller',
    use_logit = use_logit,
    is_vit = is_vit, 
    # teacher_pretrained = 'work_dirs/assets/vit-s-88.29.pth',
    teacher_pretrained = 'work_dirs/assets/mixer-b16-87.43.pth',
    teacher_cfg = 'configs/mlp_mixer/mlp-mixer-base-p16_64xb64_cifar.py',
    student_cfg ='configs/mobilenet_v2/mobilenet-v2_cifar100.py',
    distill_cfg = [
        dict(methods=[
            dict(type='UnifiedFreqDecoupleLoss',  
                 name='loss_ufdl',
                 use_this=True,
                 alpha_jfd=[0.03, 0.035, 0.003, 0.0045],  # 权重比例
                 student_dims=1280,
                 teacher_dims=768,
                 query_hw=(14,14),
                 pos_hw=(14,14),
                 window_shapes=(1, 1),
                 pos_dims=768,
                 self_query=True,
                 softmax_scale=[5., 5.],
                 dis_freq='high',
                 num_heads=32
                 )
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