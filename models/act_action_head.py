import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ──────────────────────────────────────────────────────────────────────────────
# Attention 基类（为 Deformable Attention 预留接口）
# ──────────────────────────────────────────────────────────────────────────────


class AttentionBase(nn.Module):
    """注意力机制基类，为 Deformable Attention 预留接口。"""

    def forward(self, query, key, value, **kwargs):
        raise NotImplementedError


class MultiHeadAttention(AttentionBase):
    """标准 Multi-Head Attention（当前使用）。"""

    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, **kwargs):
        B, N, D = query.shape
        M = key.shape[1]
        q = self.q_proj(query).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out), attn


class DeformableAttention(AttentionBase):
    """Deformable Attention（预留接口，暂不实现）。

    后续实现方案：
      1. 纯 PyTorch 轻量实现（推荐，保持代码整洁）
      2. 从 Deformable DETR 提取独立 CUDA 算子
      3. 使用 ms_deform_attn 库（需编译）
    """

    def __init__(self, dim, n_heads, n_points=4, dropout=0.1):
        super().__init__()
        raise NotImplementedError(
            "DeformableAttention is a placeholder. "
            "Implement with pure PyTorch or CUDA extension when needed."
        )


# ──────────────────────────────────────────────────────────────────────────────
# ACT 核心组件
# ──────────────────────────────────────────────────────────────────────────────


class ACTEncoder(nn.Module):
    """CVAE 后验编码器：q(z | observation, action)。

    输入: global_cond [B, global_cond_dim] + actions [B, horizon, action_dim]
    输出: mu [B, latent_dim], logvar [B, latent_dim]
    """

    def __init__(
        self,
        global_cond_dim=64,
        action_dim=2,
        horizon=16,
        latent_dim=32,
        dim=256,
        enc_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        action_flat_dim = horizon * action_dim
        input_dim = global_cond_dim + action_flat_dim

        layers = [nn.Linear(input_dim, dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(enc_layers - 1):
            layers.extend([nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout)])
        self.mlp = nn.Sequential(*layers)
        self.mu_head = nn.Linear(dim, latent_dim)
        self.logvar_head = nn.Linear(dim, latent_dim)

    def forward(self, global_cond, actions):
        B = actions.shape[0]
        action_flat = actions.view(B, -1)
        x = torch.cat([global_cond, action_flat], dim=-1)
        x = self.mlp(x)
        return self.mu_head(x), self.logvar_head(x)


class ACTPrior(nn.Module):
    """VAE 先验网络：p(z | observation)。

    输入: global_cond [B, global_cond_dim]
    输出: mu_prior [B, latent_dim], logvar_prior [B, latent_dim]
    """

    def __init__(
        self, global_cond_dim=64, latent_dim=32, dim=256, enc_layers=4, dropout=0.1
    ):
        super().__init__()
        layers = [nn.Linear(global_cond_dim, dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(enc_layers - 1):
            layers.extend([nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout)])
        self.mlp = nn.Sequential(*layers)
        self.mu_head = nn.Linear(dim, latent_dim)
        self.logvar_head = nn.Linear(dim, latent_dim)

    def forward(self, global_cond):
        x = self.mlp(global_cond)
        return self.mu_head(x), self.logvar_head(x)


class TransformerDecoderBlock(nn.Module):
    """单层 Transformer Decoder Block。

    结构:
      1. Self-Attention: queries 互相注意
      2. Cross-Attention: queries 注意 context
      3. FFN: 2 层 MLP
    """

    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(dim, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, queries, context):
        # Self-Attention
        sa_out, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + sa_out)
        # Cross-Attention
        ca_out, _ = self.cross_attn(queries, context, context)
        queries = self.norm2(queries + ca_out)
        # FFN
        ffn_out = self.ffn(queries)
        queries = self.norm3(queries + ffn_out)
        return queries


class ACTDecoder(nn.Module):
    """Transformer 解码器：从 latent z + global_cond 生成 action chunk。

    使用可学习的 action query tokens，通过 N 层 Transformer Decoder Block
    逐步 refine 为 action 序列。
    """

    def __init__(
        self,
        global_cond_dim=64,
        latent_dim=32,
        action_dim=2,
        n_action_steps=8,
        dim=256,
        n_heads=8,
        dec_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.action_queries = nn.Parameter(torch.randn(n_action_steps, dim))
        self.z_proj = nn.Linear(latent_dim, dim)
        self.cond_proj = nn.Linear(global_cond_dim, dim)
        self.layers = nn.ModuleList(
            [TransformerDecoderBlock(dim, n_heads, dropout) for _ in range(dec_layers)]
        )
        self.action_head = nn.Linear(dim, action_dim)

    def forward(self, global_cond, z):
        B = global_cond.shape[0]
        # 构建 context: concat(global_cond, z) 后投影到 dim
        cond = self.cond_proj(global_cond)  # [B, dim]
        z_emb = self.z_proj(z)  # [B, dim]
        context = cond + z_emb  # [B, dim]
        context = context.unsqueeze(1)  # [B, 1, dim]

        # 扩展可学习 queries 到 batch
        queries = self.action_queries.unsqueeze(0).expand(
            B, -1, -1
        )  # [B, n_action_steps, dim]

        # Transformer Decoder 层
        for layer in self.layers:
            queries = layer(queries, context)

        # 投影到 action 空间
        return self.action_head(queries)  # [B, n_action_steps, action_dim]


# ──────────────────────────────────────────────────────────────────────────────
# ACT Action Head（主类）
# ──────────────────────────────────────────────────────────────────────────────


class ACTActionHead(nn.Module):
    """Action Chunking Transformer (ACT) Action Head.

    架构:
      Encoder (CVAE 后验) + Prior (VAE 先验) + Decoder (Transformer)

    训练:
      loss = head.compute_loss(global_cond, action)
      其中: action [B, horizon, action_dim] 已归一化到 [-1, 1]

    推理:
      action_chunk = head.generate(global_cond, batch_size)
      返回: [B, n_action_steps, action_dim]（仍在 [-1, 1] 空间）
    """

    def __init__(
        self,
        action_dim=2,
        horizon=16,
        n_action_steps=4,
        global_cond_dim=128,
        dim=256,
        latent_dim=32,
        enc_layers=4,
        dec_layers=4,
        n_heads=8,
        kl_weight=10.0,
        dropout=0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.kl_weight = kl_weight

        self.encoder = ACTEncoder(
            global_cond_dim=global_cond_dim,
            action_dim=action_dim,
            horizon=horizon,
            latent_dim=latent_dim,
            dim=dim,
            enc_layers=enc_layers,
            dropout=dropout,
        )
        self.prior = ACTPrior(
            global_cond_dim=global_cond_dim,
            latent_dim=latent_dim,
            dim=dim,
            enc_layers=enc_layers,
            dropout=dropout,
        )
        self.decoder = ACTDecoder(
            global_cond_dim=global_cond_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            n_action_steps=n_action_steps,
            dim=dim,
            n_heads=n_heads,
            dec_layers=dec_layers,
            dropout=dropout,
        )

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
        """KL(q || p) = 0.5 * sum(1 + logvar_q - logvar_p - (var_q + (mu_q-mu_p)^2) / var_p)"""
        kl = 0.5 * (
            logvar_p
            - logvar_q
            + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
            - 1
        )
        return kl.sum(dim=-1).mean()

    def compute_loss(self, global_cond, action):
        """计算 ACT 训练损失。

        Args:
            global_cond: [B, global_cond_dim]  (Mamba 输出)
            action:      [B, horizon, action_dim]  (已归一化到 [-1, 1])

        Returns:
            loss = MSE_loss + kl_weight * KL_divergence
        """
        # 1. 编码器: q(z | obs, action)
        mu_q, logvar_q = self.encoder(global_cond, action)

        # 2. 先验: p(z | obs)
        mu_p, logvar_p = self.prior(global_cond)

        # 3. Reparameterize: z ~ q(z | obs, action)
        z = self.reparameterize(mu_q, logvar_q)

        # 4. 解码器: 从 z 重建 action
        action_pred = self.decoder(global_cond, z)

        # 5. MSE 重建损失（只计算前 n_action_steps）
        action_target = action[:, : self.n_action_steps]
        mse_loss = F.mse_loss(action_pred, action_target)

        # 6. KL 散度损失
        kl_loss = self.kl_divergence(mu_q, logvar_q, mu_p, logvar_p)

        return mse_loss + self.kl_weight * kl_loss

    @torch.no_grad()
    def generate(self, global_cond, batch_size=1):
        """生成 action chunk（推理模式）。

        Args:
            global_cond: [B, global_cond_dim]
            batch_size: 生成数量

        Returns:
            [B, n_action_steps, action_dim]（仍在 [-1, 1] 空间）
        """
        # 1. 先验: p(z | obs)
        mu_p, logvar_p = self.prior(global_cond)

        # 2. 确定性采样: z = mu_p（推理时不用随机采样）
        z = mu_p

        # 3. 解码器生成完整 horizon
        action_pred = self.decoder(global_cond, z)  # [B, n_action_steps, action_dim]
        return action_pred

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
