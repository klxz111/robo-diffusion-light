import torch
import torch.nn as nn
from mamba_ssm import Mamba


class RMSNorm(nn.Module):
    """Pre-RMSNorm: 比 LayerNorm 更轻量，深层网络更稳定。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class SerialHybridBlock(nn.Module):
    """串行混合块: MHA (全局空间注意力) → Mamba×4 (时序推演)。

    结构:
      x → RMSNorm → MHA → +x → RMSNorm → Mamba×4 → +x
    """

    def __init__(self, d_model=512, n_heads=8, d_state=64):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm2 = RMSNorm(d_model)
        self.temporal_mamba = nn.Sequential(
            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2),
            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2),
            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2),
            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2),
        )

    def forward(self, x):
        # MHA (全局空间注意力)
        x_norm = self.norm1(x)
        attn_out, _ = self.spatial_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Mamba×4 (时序推演)
        x_norm = self.norm2(x)
        mamba_out = self.temporal_mamba(x_norm)
        x = x + mamba_out

        return x


class ToolinitModel(nn.Module):
    def __init__(self, d_model=512, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [SerialHybridBlock(d_model) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
