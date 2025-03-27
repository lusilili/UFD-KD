_base_ = [
    '../resnet/resnet18_8xb16_cifar100.py'
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
    teacher_pretrained = 'work_dirs/assets/vit_small_patch16_224_cifar100.pth',
    teacher_cfg = 'configs/vision_transformer/vit-small_p16-4xb256_cifar100.py',
    student_cfg ='configs/vision_transformer/vit-small_p16-4xb256_cifar100.py',
    distill_cfg = [
        dict(methods=[
            dict(type='UnifiedFreqDecoupleLoss',  
                 name='loss_ufdl',
                 use_this=True,
                 alpha_jfd=[0.03, 0.065, 0.003, 0.0065],  # 权重比例
                 student_dims=384,
                 teacher_dims=768,
                 query_hw=(7,7),
                 pos_hw=(7,7),
                 window_shapes=(1, 1),
                 pos_dims=384,
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


