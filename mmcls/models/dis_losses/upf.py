import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from mmcls.registry import MODELS
import torch.fft as fft
from .attention import MultiheadPosAttention, WindowMultiheadPosAttention
from mmengine.model.weight_init import trunc_normal_
from ..utils import MultiheadAttention, resize_pos_embed, to_2tuple
from mmcv.cnn.bricks.transformer import FFN, PatchMerging


# frequency domin filtered in h*w decouple
@MODELS.register_module()
class UnifiedFreqDecoupleLoss(nn.Module):
    def __init__(self,
                 name,
                 use_this,
                 alpha_jfd,
                 student_dims,
                 teacher_dims,
                 query_hw,
                 pos_hw,
                 pos_dims,
                 window_shapes=(1,1),
                 self_query=True,
                 softmax_scale=1.,
                 dis_freq='high',
                 num_heads=8,
                 ):
        super(UnifiedFreqDecoupleLoss, self).__init__()
        self.dis_freq = dis_freq
        self.self_query = self_query
        self.alpha_jfd = alpha_jfd
        
        self.projector_0 = AttentionProjector(student_dims, teacher_dims, query_hw, pos_hw, pos_dims, window_shapes=window_shapes, self_query=self_query, softmax_scale=softmax_scale[0], num_heads=num_heads)
        self.projector_1 = AttentionProjector(student_dims, teacher_dims, query_hw, pos_hw, pos_dims, window_shapes=window_shapes, self_query=self_query, softmax_scale=softmax_scale[1], num_heads=num_heads)
        # # MGD
        # self.projector_0 = self._build_mgd_projector(student_dims, teacher_dims)
        # self.projector_1 = self._build_mgd_projector(student_dims, teacher_dims)
        
        # 权重超参
        self.alpha_dc = self.alpha_jfd[0]
        self.alpha_ac = self.alpha_jfd[1]
        self.alpha_dc_1d = self.alpha_jfd[2]
        self.alpha_ac_1d = self.alpha_jfd[3]

    # MGD
    # def _build_mgd_projector(self, in_dims, out_dims):
    #     """MGD Projector"""
    #     return nn.Sequential(
    #         # 通道对齐层（当维度不匹配时）
    #         nn.Conv2d(in_dims, out_dims, 1) if in_dims != out_dims else nn.Identity(),
    #         # MGD核心生成器结构
    #         nn.Conv2d(out_dims, out_dims, 3, padding=1),
    #         nn.BatchNorm2d(out_dims),  # 新增BN层提升稳定性
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(out_dims, out_dims, 3, padding=1),
    #         nn.BatchNorm2d(out_dims)   # 输出归一化
    #    )
    
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
        # print(preds_S.shape)
        # print(preds_T.shape)
    
        # # 获取学生特征的空间尺寸 (H, W)
        # output_size = (preds_S.size(2), preds_S.size(3))  # 假设preds_S形状为 [B, C, H, W]
    
        # # 对教师特征自适应下采样至 H x W
        # preds_T = F.adaptive_avg_pool2d(preds_T, output_size=output_size)
        # preds_T = F.adaptive_avg_pool2d(preds_T, output_size=(7, 7))
        
        # print(preds_S.shape)
        # print(preds_T.shape)
        
        spat_loss = self.get_spat_loss(self.project_feat_spat(preds_S, query=query_s), preds_T)

        channel_loss = self.get_channel_loss(self.project_feat_channel(preds_S, query = query_s), preds_T)
        
        return spat_loss, channel_loss
        # return channel_loss

    def project_feat_spat(self, preds_S, query=None):
        preds_S = self.projector_0(preds_S, query=query)
        return preds_S

    def project_feat_channel(self, preds_S, query=None):
        preds_S = self.projector_1(preds_S, query=query)
        return preds_S
    
    # # MGD
    # def project_feat_spat(self, preds_S, query=None):
    #     preds_S = self.projector_0(preds_S)
    #     return preds_S.flatten(2)

    # def project_feat_channel(self, preds_S, query=None):
    #     preds_S = self.projector_1(preds_S)
    #     return preds_S.flatten(2)


    # 空间维度：直流交流解耦
    def get_spat_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        
        N = preds_S.shape[0]
        N, C, H, W = preds_T.shape
        device = preds_S.device
        # S:[N, H*W, C]
        # T:[N, C, H, W]
            
        # 初始化 DCT 模块
        dct = DCT(resolution=H, device=device)
    
        preds_S = preds_S.permute(0,2,1).contiguous().view(*preds_T.shape)
        
        # 对学生和教师特征进行 DCT 变换
        preds_S = dct.forward(preds_S)
        preds_T = dct.forward(preds_T)
    
        # DC 和 AC 分离
        mask = torch.zeros(preds_S.shape, device=device)
        mask_dc = mask
        mask_dc[:, :, 0, 0] = 1  # 直流分量
        mask_ac = 1 - mask_dc   # 交流分量
        
        # DC 分量
        preds_S_dc = dct.inverse(torch.mul(preds_S, mask_dc))
        preds_T_dc = dct.inverse(torch.mul(preds_T, mask_dc))

        # AC 分量
        preds_S_ac = dct.inverse(torch.mul(preds_S, mask_ac))
        preds_T_ac = dct.inverse(torch.mul(preds_T, mask_ac))
 
        # 特征归一化
        preds_S_dc = F.normalize(preds_S_dc, dim=1, p=2)
        preds_T_dc = F.normalize(preds_T_dc, dim=1, p=2)
        
        preds_S_ac = F.normalize(preds_S_ac, dim=1, p=2)
        preds_T_ac = F.normalize(preds_T_ac, dim=1, p=2)
        
        # DC 和 AC 的 MSE 损失
        dc_loss = self.alpha_dc * loss_mse(preds_S_dc, preds_T_dc) / N
        ac_loss = self.alpha_ac * loss_mse(preds_S_ac, preds_T_ac) / N

        return dc_loss, ac_loss


    def get_channel_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        # S:[N, H*W, C]
        # T:[N, C, H, W]

        N = preds_S.shape[0]
        N, C, H, W = preds_T.shape
        device = preds_S.device

        # 初始化 DCT 模块
        dct = DCT(resolution=C, device=device)

        preds_T = preds_T.permute(0, 2, 3, 1)  # [N, H, W, C]
        preds_T = preds_T.contiguous()
        preds_S = preds_S.contiguous().view(*preds_T.shape)  # [N, H, W, C]

        # 对学生和教师特征进行 DCT 变换
        preds_S = dct.forward_1d(preds_S)
        preds_T = dct.forward_1d(preds_T)
    
        # DC 和 AC 分离
        mask = torch.zeros(preds_S.shape, device=device)
        mask_dc = mask
        mask_dc[:, :, :, 0] = 1  # 直流分量
        mask_ac = 1 - mask_dc   # 交流分量
        
        # DC 分量
        preds_S_dc = dct.inverse_1d(torch.mul(preds_S, mask_dc)).contiguous().view(N, H*W, C)
        preds_T_dc = dct.inverse_1d(torch.mul(preds_T, mask_dc)).contiguous().view(N, H*W, C)

        # AC 分量
        preds_S_ac = dct.inverse_1d(torch.mul(preds_S, mask_ac)).contiguous().view(N, H*W, C)
        preds_T_ac = dct.inverse_1d(torch.mul(preds_T, mask_ac)).contiguous().view(N, H*W, C)

        # 特征归一化
        preds_S_dc = F.normalize(preds_S_dc, dim=1, p=2)
        preds_T_dc = F.normalize(preds_T_dc, dim=1, p=2)

        preds_S_ac = F.normalize(preds_S_ac, dim=1, p=2)
        preds_T_ac = F.normalize(preds_T_ac, dim=1, p=2)
        
        # DC 和 AC 的 MSE 损失
        dc_loss = self.alpha_dc_1d * loss_mse(preds_S_dc, preds_T_dc) / N
        ac_loss = self.alpha_ac_1d * loss_mse(preds_S_ac, preds_T_ac) / N

        return dc_loss, ac_loss


class AttentionProjector(nn.Module):
    def __init__(self,
                 student_dims,
                 teacher_dims,
                 query_hw,
                 pos_hw,
                 pos_dims,
                 window_shapes=(1,1),
                 self_query=True,
                 softmax_scale=1.,
                 num_heads=8,
                 ):
        super(AttentionProjector, self).__init__()

        self.query_hw = query_hw
        self.pos_hw = pos_hw
        self.student_dims = student_dims
        self.teacher_dims = teacher_dims

        self.proj_pos = nn.Sequential(nn.Conv2d(teacher_dims, teacher_dims, 1),
                                      nn.ReLU(),
                                      )

        self.proj_student = nn.Sequential(nn.Conv2d(student_dims, student_dims, 3, stride=1, padding=1),
                                      nn.BatchNorm2d(student_dims),
                                      nn.ReLU(),
                                      )

        self.pos_embed = nn.Parameter(torch.zeros(1, student_dims, pos_hw[0], pos_hw[1]), requires_grad=True)
        self.pos_attention = WindowMultiheadPosAttention(teacher_dims, num_heads=num_heads, input_dims=student_dims, pos_dims=pos_dims, window_shapes=window_shapes, softmax_scale=softmax_scale)
        self.ffn = FFN(embed_dims=teacher_dims, feedforward_channels=teacher_dims * 4)

        self.norm = nn.LayerNorm([teacher_dims])

        if self_query:
            self.query = nn.Embedding(query_hw[0] * query_hw[1], teacher_dims)
        else:
            self.query = None

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)


    def forward(self, x, query=None):
        H, W = self.query_hw
        N = x.shape[0]

        if query is not None:
            query = query.permute(0,2,1).reshape(N, -1, H, W).contiguous()
        elif self.query is not None:
            query = self.query.weight.view(1,H,W,self.teacher_dims).permute(0,3,1,2).repeat(N,1,1,1)
        else:
            raise NotImplementedError("There is no query!")

        preds_S = self.proj_student(x) + self.pos_embed.to(x.device)
        query = self.proj_pos(query)
        query = torch.flatten(query.permute(0, 2, 3, 1), 1, 2)

        fea_S = self.pos_attention(torch.flatten(preds_S.permute(0, 2, 3, 1), 1, 2), query)
        fea_S = self.ffn(self.norm(fea_S))

        return fea_S


class DCT():
    def __init__(self, resolution, device, norm=None, bias=False):
        self.resolution = resolution
        self.norm = norm
        self.device = device

        I = torch.eye(self.resolution, device=self.device)
        
        self.forward_transform = nn.Linear(resolution, resolution, bias=bias).to(self.device)
        self.forward_transform.weight.data = self._dct(I, norm=self.norm).data.t()
        self.forward_transform.weight.requires_grad = False

        self.inverse_transform = nn.Linear(resolution, resolution, bias=bias).to(self.device)
        self.inverse_transform.weight.data = self._idct(I, norm=self.norm).data.t()
        self.inverse_transform.weight.requires_grad = False

    def _dct(self, x, norm=None):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)
        return V

    def _idct(self, X, norm=None):
        """
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
        Our definition of idct is that idct(dct(x)) == x
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the inverse DCT-II of the signal over the last dimension
        """

        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]
        return x.view(*x_shape)

    def forward(self, x):
        X1 = self.forward_transform(x)
        X2 = self.forward_transform(X1.transpose(-1, -2))
        return X2.transpose(-1, -2)

    def inverse(self, x):
        X1 = self.inverse_transform(x)
        X2 = self.inverse_transform(X1.transpose(-1, -2))
        return X2.transpose(-1, -2)
    
    def forward_1d(self, x):
        X1 = self.forward_transform(x)
        return X1

    def inverse_1d(self, x):
        X1 = self.inverse_transform(x)
        return X1
