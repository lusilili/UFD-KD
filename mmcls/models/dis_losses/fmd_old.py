import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from mmcls.registry import MODELS
import torch.fft as fft
from .attention import MultiheadPosAttention, WindowMultiheadPosAttention
from mmengine.model.weight_init import trunc_normal_
from ..utils import MultiheadAttention, resize_pos_embed, to_2tuple
from mmcv.cnn.bricks.transformer import FFN, PatchMerging



@MODELS.register_module()
class FreqMaskingDistillLoss(nn.Module):

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
                 softmax_scale=1.,
                 dis_freq='high'
                 ):
        super(FreqMaskingDistillLoss, self).__init__()
        self.alpha_jfd = alpha_jfd
        self.dis_freq = dis_freq

        self.projector_0 = AttentionProjector(student_dims, teacher_dims, hw_shapes, pos_dims, window_shapes=window_shapes, self_query=self_query, softmax_scale=softmax_scale)
        self.projector_1 = AttentionProjector(student_dims, teacher_dims, hw_shapes, pos_dims, window_shapes=window_shapes, self_query=self_query, softmax_scale=softmax_scale)

    def forward(self,
                preds_S,
                preds_T,
                query_s=None,
                query_f=None,
                ):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """

        preds_S_spat =  self.project_feat_spat(preds_S, query=query_s)
        preds_S_freq =  self.project_feat_freq(preds_S, query=query_f)

        spat_loss = self.get_spat_loss(preds_S_spat, preds_T)
        freq_loss = self.get_freq_loss(preds_S_freq, preds_T)

        # c_freq_loss = self.get_channel_freq_loss(preds_S, preds_T)*self.alpha_jfd[2]

        return spat_loss, freq_loss

    def project_feat_spat(self, preds_S, query=None):
        preds_S = self.projector_0(preds_S, query=query)

        return preds_S

    def project_feat_freq(self, preds_S, query=None):
        preds_S = self.projector_1(preds_S, query=query)

        return preds_S

    def get_spat_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')

        N = preds_S.shape[0]
        preds_S = preds_S.permute(0,2,1).contiguous().view(*preds_T.shape)

        preds_S = F.normalize(preds_S, dim=1)
        preds_T = F.normalize(preds_T, dim=1)

        dis_loss_arch_st = loss_mse(preds_S, preds_T)/N 
        dis_loss_arch_st = dis_loss_arch_st * self.alpha_jfd[0]

        return dis_loss_arch_st


    def get_freq_loss(self, preds_S, preds_T):
        N, C, H, W = preds_T.shape

        preds_S = preds_S.permute(0,2,1).contiguous().view(*preds_T.shape)
        device = preds_S.device
 
        preds_S = F.normalize(preds_S, dim=1, p=2)
        preds_T = F.normalize(preds_T, dim=1, p=2)

        preds_S_freq = fft.fftn(preds_S, s=(4*H, 4*W),dim=(2,3))
        preds_T_freq = fft.fftn(preds_T, s=(4*H, 4*W),dim=(2,3))

        mat = self.get_freq_mask((N,C,4*H,4*W), mode=self.dis_freq).to(device)

        preds_S_freq = torch.mul(preds_S_freq, mat)
        preds_T_freq = torch.mul(preds_T_freq, mat)

        loss_matrix = preds_T_freq - preds_S_freq

        real = torch.real(loss_matrix)
        imag = torch.imag(loss_matrix)

        dis_loss = torch.mul(real, real).sum() + torch.mul(imag, imag).sum()

        dis_loss = dis_loss * self.alpha_jfd[1]

        return dis_loss


    def get_channel_freq_loss(self, preds_S, preds_T):
        N, C, H, W = preds_T.shape

        device = preds_S.device

        preds_S = self.projector_1(preds_S)
        preds_S = preds_S.permute(0,2,1).contiguous().view(N, C, H, W)

        # preds_S = preds_S.view(N, C, H*W)
        # preds_T = preds_T.view(N, C, H*W)

        # preds_S = F.normalize(preds_S, dim=2, p=2).view(N, C, H, W)
        # preds_T = F.normalize(preds_T, dim=2, p=2).view(N, C, H, W)

        # preds_S = F.avg_pool2d(preds_S, kernel_size=7)
        # preds_T = F.avg_pool2d(preds_T, kernel_size=7)   

        preds_S = F.normalize(preds_S, dim=1, p=2)
        preds_T = F.normalize(preds_T, dim=1, p=2)

        preds_S_freq = fft.fftn(preds_S, dim=(1))
        preds_T_freq = fft.fftn(preds_T, dim=(1))

        ############### mask low freq ################
        preds_S_freq[:, 3*C//8:5*C//8,:,:] = 0
        preds_T_freq[:, 3*C//8:5*C//8,:,:] = 0
        ##############################################


        ############### mask high freq ###############
        # preds_S_freq[:,:C//8,:,:] = 0
        # preds_S_freq[:,-C//8-1:,:,:] = 0
        # preds_T_freq[:,:C//8,:,:] = 0
        # preds_T_freq[:,-C//8-1:,:,:] = 0
        ##############################################

        loss_matrix = preds_T_freq -  preds_S_freq

        real = torch.real(loss_matrix)
        imag = torch.imag(loss_matrix)

        dis_loss = torch.mul(real, real).sum() + torch.mul(imag, imag).sum()

        return dis_loss
    

    def get_freq_mask(self, shape, mode='low'):
        N, C, H, W = shape

        if mode == 'low':
            mat = torch.ones(N, 1, H, W)
            mat[:,:,0:H//4,0:H//4] = 0
            mat[:,:,0:H//4,-H//4-1:-1] = 0
            mat[:,:,-H//4-1:-1,0:H//4] = 0
            mat[:,:,-H//4-1:-1,-H//4-1:-1] = 0
            return mat
        elif mode == 'high':
            mat = torch.ones(N, 1, H, W)
            mat[:,:,0:2*H//5,0:2*H//5] = 0
            mat[:,:,0:2*H//5,-2*H//5-1:-1] = 0
            mat[:,:,-2*H//5-1:-1,0:2*H//5] = 0
            mat[:,:,-2*H//5-1:-1,-2*H//5-1:-1] = 0
            return 1-mat
        elif mode == 'all':
            mat = torch.ones(N, 1, H, W)
            return mat
        else:
            assert 0 == 1



class AttentionProjector(nn.Module):
    def __init__(self,
                 student_dims,
                 teacher_dims,
                 hw_dims,
                 pos_dims,
                 window_shapes=(1,1),
                 self_query=True,
                 softmax_scale=1.,
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

        if query is not None:
            pos_emb = query.permute(0,2,1).reshape(N, -1, H, W).contiguous()
        elif self.query is not None:
            pos_emb = self.query.weight.view(1,H,W,self.teacher_dims).permute(0,3,1,2).repeat(N,1,1,1)
        else:
            raise NotImplementedError("There is no query!")

        preds_S = self.proj_student(x) + self.pos_embed.to(x.device)
        pos_emb = self.proj_pos(pos_emb)
        pos_emb = torch.flatten(pos_emb.permute(0, 2, 3, 1), 1, 2)

        fea_S = self.pos_attention(torch.flatten(preds_S.permute(0, 2, 3, 1), 1, 2), pos_emb)
        fea_S = self.ffn(self.norm(fea_S))

        return fea_S