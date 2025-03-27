import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from mmcls.registry import MODELS
from .attention import MultiheadPosAttention, WindowMultiheadPosAttention
from mmengine.model.weight_init import trunc_normal_
from ..utils import MultiheadAttention, resize_pos_embed, to_2tuple
from mmcv.cnn.bricks.transformer import FFN, PatchMerging



@MODELS.register_module()
class JFDLossv2(nn.Module):

    def __init__(self,
                 name,
                 use_this,
                 alpha_jfd,
                 student_dims,
                 teacher_dims,
                 hw_shapes,
                 pos_dims,
                 window_shapes=(1,1),
                 self_query=True,
                 softmax_scale=1.
                 ):
        super(JFDLossv2, self).__init__()
        self.alpha_jfd = alpha_jfd
        self.projector = AttentionProjector(student_dims, teacher_dims, hw_shapes, pos_dims, window_shapes=window_shapes, self_query=self_query, softmax_scale=softmax_scale)

        


    def forward(self,
                preds_S,
                preds_T,
                mask=None,
                ):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        loss_mse = nn.MSELoss(reduction='none')

        N = preds_S.shape[0]
        device = preds_S.device

        preds_S = F.normalize(preds_S, dim=2)
        preds_T = F.normalize(torch.flatten(preds_T.permute(0, 2, 3, 1), 1, 2), dim=2)

        dis_loss_arch_st = loss_mse(preds_S, preds_T)

        if mask is not None:
            mask = mask.permute(0,2,3,1).contiguous().view(N, 49, -1).to(device)
            dis_loss_arch_st = torch.mul(dis_loss_arch_st, mask).sum()/N * self.alpha_jfd
        else:
            dis_loss_arch_st = dis_loss_arch_st.sum()/N * self.alpha_jfd

        return dis_loss_arch_st


    def project_feat(self, preds_S, query=None):
        out = self.projector(preds_S, query)

        return out

    def mix_feat(self, preds_S1, preds_S2, ratio=0.50):
        N, HW, C = preds_S1.shape
        device = preds_S1.device


        mat = torch.rand((N,HW,1)).to(device)
        mat = torch.where(mat < 1-ratio, 0, 1).to(device)

        preds_S1 = torch.mul(preds_S1, mat)
        preds_S2 = torch.mul(preds_S2, 1-mat)

        out_feat = (preds_S1 + preds_S2).to(device)

        return out_feat


class AttentionProjector(nn.Module):
    def __init__(self,
                 student_dims,
                 teacher_dims,
                 hw_dims,
                 pos_dims,
                 window_shapes=(1,1),
                 self_query=True,
                 softmax_scale=1.0,
                 ):
        super(AttentionProjector, self).__init__()

        self.hw_dims = hw_dims
        self.student_dims = student_dims
        self.teacher_dims = teacher_dims

        self.proj_pos = nn.Sequential(nn.Conv2d(teacher_dims, teacher_dims, 1),
                                      nn.ReLU(),
                                      )

        self.proj_student = nn.Sequential(nn.Conv2d(student_dims, student_dims, 3, stride=1, padding=1),
                                      nn.BatchNorm2d(student_dims),
                                      nn.ReLU())

        self.pos_embed = nn.Parameter(torch.zeros(1, student_dims, hw_dims[0], hw_dims[1]), requires_grad=True)
        self.pos_attention = WindowMultiheadPosAttention(teacher_dims, num_heads=8, input_dims=student_dims, pos_dims=pos_dims, window_shapes=window_shapes, softmax_scale=softmax_scale)
        self.ffn = FFN(embed_dims=teacher_dims, feedforward_channels=teacher_dims * 4)

        self.norm = nn.LayerNorm([teacher_dims])

        if self_query:
            self.query = nn.Embedding(hw_dims[0] * hw_dims[1], teacher_dims)
        else:
            self.query = None

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)


    def forward(self, x, query=None):
        H, W = self.hw_dims
        N = x.shape[0]
        if self.query is not None:
            pos_emb = self.query.weight.view(1,H,W,self.teacher_dims).permute(0,3,1,2).repeat(N,1,1,1)
        elif query is not None:
            pos_emb = query.permute(0,2,1).reshape(N, -1, H, W).contiguous()
        else:
            raise NotImplementedError("There is no query!")

        preds_S = self.proj_student(x) + self.pos_embed.to(x.device)
        pos_emb = self.proj_pos(pos_emb)
        pos_emb = torch.flatten(pos_emb.permute(0, 2, 3, 1), 1, 2)

        fea_S = self.pos_attention(torch.flatten(preds_S.permute(0, 2, 3, 1), 1, 2), pos_emb)
        fea_S = self.ffn(fea_S) + fea_S
        fea_S = self.norm(fea_S) 

        return fea_S