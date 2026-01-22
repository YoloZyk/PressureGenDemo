# motion_generation/lib/model/ddpm.py

import torch
import torch.nn as nn
from lib.model.unet1d import UNet1D


class DDPM(nn.Module):
    def __init__(self, model: UNet1D, beta_schedule='linear', timesteps=1000):
        super().__init__()
        self.model = model  # 噪声预测器 (UNet1D)
        self.timesteps = timesteps

        # ----------------------------------------------------
        # 1. 定义 Beta 调度和相关参数
        # ----------------------------------------------------
        
        # 1.1. 计算 Betas (噪声水平)
        if beta_schedule == 'linear':
            betas = self._linear_beta_schedule(timesteps)
        else:
            raise NotImplementedError(f"不支持的 beta_schedule: {beta_schedule}")

        # 将所有参数注册为 buffer (不会被训练，但会保存到 state_dict)
        self.register_buffer('betas', betas)
        
        # 1.2. 计算 Alpha 相关的参数
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # 累积乘积: \bar{\alpha}_t
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # 方便计算的根号形式
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)  # 1/sqrt(alpha_t)

        # 后验均值计算所需的参数 (反向过程)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # 注册 buffers
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('posterior_variance', posterior_variance)
        
        # ----------------------------------------------------

    def _linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        """线性 Beta 调度"""
        return torch.linspace(start, end, timesteps)

    def get_index_from_t(self, variances, t, x_shape):
        """从 (T,) 维度的张量中，根据时间步 t (B,) 提取对应的方差/系数，并重塑至 (B, 1, 1...)"""
        B = t.shape[0]
        out = variances.gather(-1, t) # 从 variances 中提取 t 对应的值

        # 重塑维度以匹配 x 的形状 (B, C) -> (B, C)
        # 对于我们的姿态数据 (B, 72)， C=72，所以只需要重塑到 (B, 1)
        return out.reshape(B, *([1] * (len(x_shape) - 1)))


    # ----------------------------------------------------
    # 2. 前向扩散 (加噪)
    # ----------------------------------------------------
    def forward_diffusion(self, x_start, t, noise=None):
        """
        前向过程：计算 x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        
        Args:
            x_start: 初始数据 x_0 (B, 72)
            t: 时间步 (B,)
            noise: 用于加噪的噪声，如果为 None 则随机生成
        
        Returns:
            x_t: t时刻加噪后的数据 (B, 72)
            noise: 实际使用的噪声 (B, 72)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # 提取 sqrt(alpha_bar_t) 和 sqrt(1 - alpha_bar_t)
        sqrt_alpha_bar_t = self.get_index_from_t(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_bar_t = self.get_index_from_t(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        # 公式实现
        x_t = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise

    # ----------------------------------------------------
    # 3. 损失计算 (训练过程)
    # ----------------------------------------------------
    def forward(self, x_start):
        """
        训练时的前向传播，用于计算损失。
        
        Args:
            x_start: 批量原始数据 (归一化后的姿态) (B, 72)
        
        Returns:
            loss: 均方误差 (MSE) 损失
        """
        B = x_start.shape[0]
        
        # 1. 随机采样时间步 t
        t = torch.randint(0, self.timesteps, (B,), device=x_start.device).long()
        
        # 2. 随机生成噪声
        noise = torch.randn_like(x_start)
        
        # 3. 计算 x_t 和 t 时刻的真实噪声
        x_t, _ = self.forward_diffusion(x_start, t, noise)
        
        # 4. 噪声预测网络预测噪声
        predicted_noise = self.model(x_t, t)
        
        # 5. 计算损失：预测噪声和真实噪声的均方误差 (MSE)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        return loss

    # ----------------------------------------------------
    # 4. 采样 (生成过程)
    # ----------------------------------------------------
    @torch.no_grad()
    def sample(self, sample_shape, device, return_intermediates=False, save_interval=None):
        """
        逆向过程：从纯噪声开始，迭代去噪 T 步，生成新的姿态数据。

        Args:
            sample_shape: 要生成的样本形状 (B, 72)
            device: 运行设备
            return_intermediates: 是否返回中间状态
            save_interval: 保存中间状态的间隔（timesteps），如果为None则不保存

        Returns:
            如果 return_intermediates=False:
                x_0: 最终生成的归一化姿态 (B, 72)
            如果 return_intermediates=True:
                (x_0, intermediates): 最终姿态和中间状态列表
                intermediates: List[(timestep, x_t)] 包含时间步和对应的状态
        """
        # 从纯噪声开始 x_T ~ N(0, I)
        x = torch.randn(sample_shape, device=device)

        # 存储中间状态
        intermediates = []
        if return_intermediates:
            # 保存初始噪声状态
            intermediates.append((self.timesteps, x.clone().cpu()))

        # 从 T-1 步迭代到 0 步
        for t in reversed(range(0, self.timesteps)):
            t_tensor = torch.full((sample_shape[0],), t, device=device, dtype=torch.long)

            # 1. 预测噪声
            predicted_noise = self.model(x, t_tensor)

            # 2. 提取当前时刻的系数
            beta_t = self.get_index_from_t(self.betas, t_tensor, x.shape)
            sqrt_one_minus_alpha_bar_t = self.get_index_from_t(self.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape)
            sqrt_recip_alpha_t = self.get_index_from_t(self.sqrt_recip_alphas, t_tensor, x.shape)

            # 3. 计算均值 (mu_t-1)
            # DDPM公式: μ_t = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * predicted_noise)
            mean = sqrt_recip_alpha_t * (x - beta_t * predicted_noise / sqrt_one_minus_alpha_bar_t)

            # 4. 添加噪声项
            if t > 0:
                variance = self.get_index_from_t(self.posterior_variance, t_tensor, x.shape)
                noise = torch.randn_like(x)
                # x_{t-1} = \mu_{t-1} + \sigma_{t-1} * z
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean # t=0 时不加噪声，直接取均值作为最终输出

            # 保存中间状态
            if return_intermediates:
                if save_interval is None:
                    # 保存所有状态
                    intermediates.append((t, x.clone().cpu()))
                elif t % save_interval == 0 or t == 0:
                    # 按间隔保存
                    intermediates.append((t, x.clone().cpu()))

        # 将输出限制在 [-1, 1] 附近（可选，取决于您的归一化范围）
        # x = x.clamp(-1., 1.)

        if return_intermediates:
            return x, intermediates
        return x
    
    