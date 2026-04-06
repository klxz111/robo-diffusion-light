"""
ViT Backbone for V4 Training Script.

Uses DINO v1 vits8 (frozen) as the vision backbone.
Outputs token sequence compatible with SpatialSoftmax.
"""

import torch
import torch.nn as nn


class ViTBackbone(nn.Module):
    """ViT backbone using DINO v1 vits8 (frozen).

    Args:
        embed_dim: Output embedding dimension (default: 512)
        n_obs: Number of observation frames (default: 4)

    Input:
        x: [B, n_obs, C, H, W] or [B * n_obs, C, H, W]

    Output:
        [B * n_obs, num_tokens, embed_dim]
    """

    def __init__(self, embed_dim=512, n_obs=4):
        super().__init__()
        self.n_obs = n_obs

        # Load DINO v1 vits8 from torch.hub
        self.dino = torch.hub.load("facebookresearch/dino:main", "dino_vits8")

        # Freeze DINO backbone
        for param in self.dino.parameters():
            param.requires_grad = False

        # DINO vits8: patch_size=8, embed_dim=384, num_heads=6
        # Input 96x96 -> 12x12 patches = 144 tokens + 1 cls token = 145 tokens
        dino_dim = 384

        # Projection: [B, num_tokens, 384] -> [B, num_tokens, 512]
        self.proj = nn.Linear(dino_dim, embed_dim)

    def forward(self, x):
        # Handle both [B, n_obs, C, H, W] and [B * n_obs, C, H, W]
        if x.ndim == 5:
            B, n_obs, C, H, W = x.shape
            x = x.view(B * n_obs, C, H, W)
        else:
            B = x.shape[0]

        # DINO forward pass (no grad since frozen)
        with torch.no_grad():
            # dino returns [B, num_tokens, 384] including cls token
            # For vits8 with 96x96 input: 144 patch tokens + 1 cls = 145
            features = self.dino.get_intermediate_layers(x, n=1)[0]

        # Remove cls token (first token), keep only patch tokens
        # [B, 145, 384] -> [B, 144, 384]
        features = features[:, 1:, :]

        # Project to target dimension
        out = self.proj(features)  # [B, 144, 512]
        return out

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
