# motion_generation/lib/model/unet1d.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    用于编码时间步 t 的标准 Transformer 位置编码
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.register_buffer('inv_freq', 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, x):
        # x 形状: (B,) 时间步索引
        sinusoid_inp = torch.einsum('i, j -> i j', x.float(), self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb

class ResBlock1D(nn.Module):
    """
    一维残差块，包含 time embedding 的融合
    """
    def __init__(self, in_channels, out_channels, time_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()

        # 时间步嵌入层
        self.time_proj = nn.Linear(time_dim, out_channels)

        # 确保输入/输出通道匹配
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        # x: (B, C_in, L) L=72, C_in=通道数

        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act1(h)

        # 融合时间嵌入：沿特征维度广播并相加
        time_emb_proj = self.time_proj(time_emb).unsqueeze(-1) # (B, C_out) -> (B, C_out, 1)
        h = h + time_emb_proj

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.act2(h)

        return h + self.residual_conv(x)


class SelfAttention1D(nn.Module):
    """
    一维自注意力模块，包含 time embedding 的融合
    """
    def __init__(self, channels, time_dim, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        # Group normalization for better stability
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)

        # Multi-head attention components
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

        # Time embedding projection
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x, time_emb):
        # x: (B, C, L)
        B, C, L = x.shape

        # Normalize input
        h = self.norm(x)

        # Add time embedding
        time_emb_proj = self.time_proj(time_emb).unsqueeze(-1)  # (B, C, 1)
        h = h + time_emb_proj

        # Compute Q, K, V
        qkv = self.qkv(h)  # (B, 3*C, L)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, L)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, num_heads, L, L)
        attn = torch.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, num_heads, L, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, L)  # (B, C, L)

        # Project and add residual
        out = self.proj(out)
        return out + x


class UNet1D(nn.Module):
    def __init__(self, pose_dim=72, base_channels=128, channel_multipliers=[1, 2, 4],
                 time_emb_dim=256, mid_structure='conv', mid_num_heads=4):
        """
        UNet1D model for 1D sequence processing

        Args:
            pose_dim: Dimension of input pose (default: 72)
            base_channels: Base number of channels (default: 128)
            channel_multipliers: Channel multipliers for each level (default: [1, 2, 4])
            time_emb_dim: Time embedding dimension (default: 256)
            mid_structure: Structure for middle layer, either 'conv' or 'attention' (default: 'conv')
            mid_num_heads: Number of attention heads for mid layer when using attention (default: 4)
        """
        super().__init__()

        self.mid_structure = mid_structure
        assert mid_structure in ['conv', 'attention'], "mid_structure must be 'conv' or 'attention'"

        # 1. Time Embedding
        self.time_mlp = nn.Sequential(
            PositionalEncoding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # 调整输入通道：姿态向量 (B, 72) -> (B, 1, 72)
        # 我们将特征维度 (72) 视为长度 L，将 1 视为通道 C
        in_channels = 1

        # 2. 编码器 (Downsampling)
        channels = [in_channels] + [base_channels * m for m in channel_multipliers]
        self.downs = nn.ModuleList()

        for i in range(len(channel_multipliers)):
            in_c = channels[i]
            out_c = channels[i+1]

            self.downs.append(nn.ModuleList([
                ResBlock1D(in_c if i == 0 else in_c, out_c, time_emb_dim), # 输入是 1, L 或 C_in, L
                nn.MaxPool1d(2) if i < len(channel_multipliers) - 1 else nn.Identity()
            ]))

        # 3. 中间层 - 根据 mid_structure 选择使用卷积或自注意力
        mid_c = channels[-1]
        if mid_structure == 'conv':
            self.mid = ResBlock1D(mid_c, mid_c, time_emb_dim)
        elif mid_structure == 'attention':
            # 确保通道数能被注意力头数整除
            assert mid_c % mid_num_heads == 0, f"mid_c ({mid_c}) must be divisible by mid_num_heads ({mid_num_heads})"
            self.mid = SelfAttention1D(mid_c, time_emb_dim, num_heads=mid_num_heads)
        
        # 4. 解码器 (Upsampling)
        self.ups = nn.ModuleList()
        reversed_channels = list(reversed(channels))

        for i in range(len(channel_multipliers)):
            in_c = reversed_channels[i]  # 来自上一层的通道数
            # 最后一层输出 base_channels，而不是 in_channels (1)
            out_c = reversed_channels[i+1] if i < len(channel_multipliers) - 1 else base_channels
            skip_c = in_c  # 跳跃连接通道数（来自对应编码器层）

            self.ups.append(nn.ModuleList([
                # ResBlock 接收拼接后的通道: in_c(来自上层) + skip_c(来自编码器)
                # 输出为 out_c 通道
                ResBlock1D(in_c + skip_c, out_c, time_emb_dim),
                # 上采样到下一层的空间尺寸
                nn.ConvTranspose1d(out_c, out_c, kernel_size=2, stride=2) if i < len(channel_multipliers) - 1 else nn.Identity(),
            ]))

        # 5. 输出层 (回到 1 个通道)
        self.out_conv = nn.Conv1d(base_channels, in_channels, kernel_size=1)
        
    def forward(self, x, t):
        # import pdb; pdb.set_trace()
        # x: (B, 72) 归一化姿态，t: (B,) 时间步索引
        x = x.unsqueeze(1) # (B, 1, 72)

        # 1. Time Embedding
        time_emb = self.time_mlp(t)
        
        # 2. 编码器
        skips = []
        for resblock, downsample in self.downs:
            x = resblock(x, time_emb)
            skips.append(x)
            x = downsample(x)

        # 3. 中间层
        x = self.mid(x, time_emb)
        
        # 4. 解码器
        for i, (resblock, upsample) in enumerate(self.ups):
            skip = skips.pop()

            # 跳跃连接
            # 检查维度是否匹配，如果 MaxPool 导致了奇数/偶数长度不匹配，需要裁剪
            if x.shape[-1] != skip.shape[-1]:
                x = nn.functional.pad(x, (0, skip.shape[-1] - x.shape[-1]))

            # 拼接跳跃连接
            x = torch.cat((x, skip), dim=1) # 沿通道维度拼接

            # 处理拼接后的特征
            x = resblock(x, time_emb)

            # 上采样到下一层的空间尺寸（最后一层不上采样）
            if i < len(self.ups) - 1:
                x = upsample(x) # (B, C, L) -> (B, C, L*2)

        # 5. 输出层
        x = self.out_conv(x) # (B, 1, 72)
        
        return x.squeeze(1) # (B, 72)


if __name__ == "__main__":
    # 测试 UNet1D 模型
    print("Testing UNet1D with conv mid structure...")
    model_conv = UNet1D(mid_structure='conv')

    x = torch.randn(4, 72) # 模拟输入 (B, 72)
    t = torch.randint(0, 1000, (4,)) # 模拟时间步索引 (B,)

    output_conv = model_conv(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape (conv): {output_conv.shape}")
    print(f"Expected output shape: (4, 72)")
    print(f"Test passed (conv): {output_conv.shape == torch.Size([4, 72])}")

    print("\nTesting UNet1D with attention mid structure...")
    model_attn = UNet1D(mid_structure='attention', base_channels=128, mid_num_heads=8)

    output_attn = model_attn(x, t)
    print(f"Output shape (attention): {output_attn.shape}")
    print(f"Test passed (attention): {output_attn.shape == torch.Size([4, 72])}")

    # 统计参数量
    conv_params = sum(p.numel() for p in model_conv.parameters())
    attn_params = sum(p.numel() for p in model_attn.parameters())
    print(f"\nParameters (conv): {conv_params:,}")
    print(f"Parameters (attention): {attn_params:,}")
    print(f"Difference: {abs(attn_params - conv_params):,}")



