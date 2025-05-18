import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['UnifiedFrequencyDecouplingLoss', 'DCT']

class UnifiedFrequencyDecouplingLoss(nn.Module):
    def __init__(self, 
                student_channels=512, 
                teacher_channels=2048, 
                tau_ocd=0.07, 
                M_ocd=16, 
                pool_size=4,
                patch_size=(4,4),
                rand_mask=True, 
                mask_ratio=0.75,
                enhance_projector=False,
                dataset='citys',
                alpha_jfd=(0.03, 0.035, 0.005, 0.0035),  # 新增频域损失权重
                ):
        
        super(UnifiedFrequencyDecouplingLoss, self).__init__()
        
        # 原OmniContrastive参数
        self.tau_ocd = tau_ocd
        self.M_ocd = M_ocd
        self.pool_size = pool_size
        self.patch_size = patch_size
        
        
        self.alpha_jfd = alpha_jfd
        # 频域解耦参数
        self.alpha_dc = self.alpha_jfd[0]
        self.alpha_ac = self.alpha_jfd[1]
        self.alpha_dc_1d = self.alpha_jfd[2]
        self.alpha_ac_1d = self.alpha_jfd[3]
        
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)

        self.generator = None
        if enhance_projector:
            self.projector_0 = EnhancedProjector(teacher_channels, teacher_channels)
            self.projector_1 = EnhancedProjector(teacher_channels, teacher_channels)
        else:
            self.projector_0 = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))
            self.projector_1 = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))

        
    def forward(self, preds_S, preds_T):
        spat_dc_loss, spat_ac_loss = self.spatial_freq_loss(self.project_feat_spat(preds_S), preds_T)
        chan_dc_loss, chan_ac_loss = self.channel_freq_loss(self.project_feat_channel(preds_S), preds_T)
        
        return spat_dc_loss, spat_ac_loss, chan_dc_loss, chan_ac_loss
    
    def project_feat_spat(self, preds_S):
        preds_S = self.align(preds_S)
        preds_S = self.projector_0(preds_S)
        return preds_S

    def project_feat_channel(self, preds_S):
        preds_S = self.align(preds_S)
        preds_S = self.projector_1(preds_S)
        return preds_S
    
    def spatial_freq_loss(self, preds_S, preds_T):
        """空间频域解耦损失"""
        loss_mse = nn.MSELoss(reduction='sum')
        
        # 获取教师特征维度[4](@ref)
        N, C, H, W = preds_T.shape  
        device = preds_S.device

        # 确保学生特征维度对齐[5](@ref)
        assert preds_S.dim() == 4, f"输入应为4D张量，当前维度：{preds_S.shape}"

        # 正确的维度转换流程[2,3](@ref)
        preds_S = preds_S.permute(0, 2, 3, 1)  # [N,C,H,W] → [N,H,W,C]
        preds_S = preds_S.contiguous().view(N, H*W, C)  # [N,H,W,C] → [N,HW,C]
        preds_S = preds_S.permute(0, 2, 1)       # [N,C,HW]
        preds_S = preds_S.view(N, C, H, W)      # 匹配教师特征形状

        # 初始化 DCT 模块[6,7](@ref)
        dct = DCT(resolution=H, device=device)
        
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


    def channel_freq_loss(self, preds_S, preds_T):
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


# Projector
class EnhancedProjector(nn.Module):
    def __init__(self, 
                in_channels=2048, 
                out_channels=2048, 
                ):
        super(EnhancedProjector, self).__init__()
        self.block_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True))
        self.block_2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True))
        self.adpator_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.adpator_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_1 = self.block_1(x)
        x_2 = self.block_2(x_1)

        x_1 = self.adpator_1(x_1)
        x_2 = self.adpator_2(x_2)

        out = (x_1 + x_2)/2.

        return out


# DCT类    
class DCT(nn.Module):
    def __init__(self, resolution, device, norm=None, bias=False):
        super().__init__()
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
