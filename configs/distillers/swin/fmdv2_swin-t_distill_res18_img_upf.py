_base_ = [
    '../../resnet/resnet18_8xb32_in1k.py'
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
# method details
model = dict(
    _delete_ = True,
    type='ClassificationDistiller',
    use_logit = use_logit,
    is_vit = is_vit,
    sync_bn=True,
    teacher_pretrained = 'work_dirs/assets/swin-t_e300_scalekd_from_swin-l.pth',
    teacher_cfg = 'configs/swin_transformer/swin-tiny_16xb64_in1k.py',
    student_cfg ='configs/resnet/resnet18_8xb32_in1k.py',
    distill_cfg = [
          dict(methods=[
            dict(type='UnifiedPathDynamicWeightFreqLoss',  
                 name='loss_updwfl',
                 use_this=True,
                 alpha_jfd=[0.03, 0.065, 0.003, 0.0065],  # 权重比例
                 student_dims=512,
                 teacher_dims=768,
                 query_hw=(7,7),
                 pos_hw=(7,7),
                 window_shapes=(1, 1),
                 pos_dims=768,
                 self_query=True,
                 softmax_scale=[5., 5.],
                 dis_freq='high',
                 num_heads=24)
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

# 