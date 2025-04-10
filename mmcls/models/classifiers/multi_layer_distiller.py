from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcls.models import build_classifier
from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample

from mmengine.config import Config
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement
from mmengine.runner.checkpoint import  load_checkpoint, _load_checkpoint, load_state_dict

@MODELS.register_module()
class MultiLayerClassificationDistiller(BaseModel, metaclass=ABCMeta):
    """Base distiller for dis_classifiers.

    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 is_vit = False,
                 use_logit = False,
                 sd = False,
                 distill_cfg = None,
                 teacher_pretrained = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):

        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type', 'mmcls.ClsDataPreprocessor')

        if train_cfg is not None and 'augments' in train_cfg:
            # Set batch augmentations by `train_cfg`
            data_preprocessor['batch_augments'] = train_cfg

        super(MultiLayerClassificationDistiller, self).__init__(
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.teacher = build_classifier((Config.fromfile(teacher_cfg)).model)
        self.teacher_pretrained = teacher_pretrained
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student = build_classifier((Config.fromfile(student_cfg)).model)

        self.distill_cfg = distill_cfg   
        self.distill_losses = nn.ModuleDict()
        if self.distill_cfg is not None:  
            for item_loc in distill_cfg:
                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    use_this = item_loss.use_this
                    if use_this:
                        self.distill_losses[loss_name] = MODELS.build(item_loss)

        self.is_vit = is_vit
        self.sd = sd
        self.use_logit = use_logit

        if 'loss_tcd' in self.distill_losses.keys():
            self.distill_losses['loss_tcd'].set_head(self.teacher.head)

    def init_weights(self):
        if self.teacher_pretrained is not None:
            load_checkpoint(self.teacher, self.teacher_pretrained, map_location='cpu')
        self.student.init_weights()

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor'):
        if mode == 'tensor':
            feats = self.student.extract_feat(inputs)
            return self.student.head(feats) if self.student.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.student.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def loss(self, inputs: torch.Tensor,
             data_samples: List[ClsDataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[ClsDataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if 'score' in data_samples[0].gt_label:
            # Batch augmentation may convert labels to one-hot format scores.
            gt_label = torch.stack([i.gt_label.score for i in data_samples])
        else:
            gt_label = torch.cat([i.gt_label.label for i in data_samples])

        fea_s = self.student.extract_feat(inputs, stage='backbone')    

        x = fea_s
        if self.student.with_neck:
            x = self.student.neck(x)
        if self.student.with_head and hasattr(self.student.head, 'pre_logits'):
            x = self.student.head.pre_logits(x)

        if self.is_vit:
            logit_s = self.student.head.layers.head(x)
        else:
            logit_s = self.student.head.fc(x)
        loss = self.student.head._get_loss(logit_s, data_samples)

        s_loss = dict()
        for key in loss.keys():
            s_loss['ori_'+key] = loss[key]

        with torch.no_grad():
            fea_t = self.teacher.extract_feat(inputs, stage='backbone')
            if self.use_logit:
                logit_t = self.teacher.head.layers.head(self.teacher.head.pre_logits(fea_t))
        

        all_keys = self.distill_losses.keys()


        if 'loss_tfd' in all_keys:
            loss_name = 'loss_tfd'
            preds_S = fea_s[-1]
            preds_T = fea_t[-1]
            # print(preds_S.shape)
            # print(preds_T.shape)
            # assert 0 == 1
            # pos_emb_1 = fea_t[0]   # feature before the first stage
            pos_emb = fea_t[-2] # feature before the last stage
            # s_loss[loss_name] = self.distill_losses[loss_name](preds_S, preds_T, pos_emb)
            s_loss[loss_name] = self.distill_losses[loss_name](preds_S, preds_T, pos_emb)
        


        if 'loss_fd_s1' in all_keys:
            loss_name = 'loss_fd_s1'
            preds_S = fea_s[0]
            preds_T = fea_t[1]
            s_loss[loss_name] = self.distill_losses[loss_name](preds_S, preds_T)

        if 'loss_fd_s2' in all_keys:
            loss_name = 'loss_fd_s2'
            preds_S = fea_s[1]
            preds_T = fea_t[2]
            s_loss[loss_name] = self.distill_losses[loss_name](preds_S, preds_T)

        if 'loss_fd_s3' in all_keys:
            loss_name = 'loss_fd_s3'
            preds_S = fea_s[2]
            preds_T = fea_t[3]
            s_loss[loss_name] = self.distill_losses[loss_name](preds_S, preds_T)

        if 'loss_fd_s4' in all_keys:
            loss_name = 'loss_fd_s4'
            preds_S = fea_s[3]
            preds_T = fea_t[4]
            s_loss[loss_name] = self.distill_losses[loss_name](preds_S, preds_T)



        if ('loss_kd' in all_keys) and self.use_logit:
            loss_name = 'loss_kd'
            ori_alpha, s_loss[loss_name] = self.distill_losses[loss_name](logit_s, logit_t)
            s_loss['ori_loss'] = ori_alpha * s_loss['ori_loss']


        return s_loss