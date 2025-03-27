_base_ = [
    '../../mlp_mixer/mlp-mixer-base-p16_64xb64_in1k.py'
]
# model settings
find_unused_parameters = True

# distillation settings
use_logit = True
is_vit = True

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
    teacher_pretrained = 'ckpt/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221226-f61cf992.pth',
    teacher_cfg = 'configs/eva/vit-b-eva.py',
    student_cfg ='configs/mlp_mixer/mlp-mixer-base-p16_64xb64_in1k.py',
    distill_cfg = [dict(methods=[dict(type='FreqMaskingDistillLoss',
                                       name='loss_fmd_s3',
                                       use_this=True,
                                       alpha_jfd=[0.02, 0.015],
                                       student_dims=768,
                                       teacher_dims=768,
                                       query_hw=(14,14),
                                       pos_hw=(14,14),
                                       pos_dims=768,
                                       self_query=True,
                                       softmax_scale=[1.0, 1.0],
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='FreqMaskingDistillLoss',
                                       name='loss_fmd_s4',
                                       use_this=True,
                                       alpha_jfd=[0.02, 0.015],
                                       student_dims=768,
                                       teacher_dims=768,
                                       query_hw=(14,14),
                                       pos_hw=(14,14),
                                       pos_dims=768,
                                       self_query=False,
                                       softmax_scale=[1.0, 1.0],
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
# /code/fanjiawei/fanjiawei/code/cls_KD_remote/configs/eva/vit-b-eva.py
# dataset_type = 'ImageNet'
# train_dataloader = dict(
#     batch_size=48,
#     num_workers=8,
#     dataset=dict(
#         type=dataset_type,
#         data_root='/data/fanjiawei/fanjiawei/dataset/imagenet',
#         data_prefix='train',
#         pipeline=train_pipeline),
#     sampler=dict(type='DefaultSampler', shuffle=True),
# )

# train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=301)
