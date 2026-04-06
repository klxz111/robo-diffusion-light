"""
Spatial Softmax module for V4.

Uses Conv2d to preserve 2D spatial topology of ViT patch tokens.

Usage:
    out = core(feat)  # [B, 576, 512]  (4 frames x 144 tokens)
    out = out.view(B, 4, 144, 512)     # Unfold time
    out = out.view(B * 4, 144, 512)    # Fold into batch
    keypoints, kp_features = spatial_softmax(out)  # [4B, 16, 2], [4B, 16, 512]
    keypoints = keypoints.view(B, 4, 16, 2)  # [B, 4, 16, 2]
    kp_features = kp_features.view(B, 4, 16, 512)  # [B, 4, 16, 512]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """Spatial Softmax: 从单帧 2D token grid 中学习 K 个关键点的空间坐标。

    输入: [B, num_tokens, token_dim]  (单帧 tokens, num_tokens = grid_h * grid_w)
    输出:
      - keypoints: [B, K, 2]  (归一化坐标 [-1, 1])
      - keypoint_features: [B, K, token_dim]  (关键点处的特征)
    """

    def __init__(self, num_keypoints=16, token_dim=512, grid_h=12, grid_w=12):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.token_dim = token_dim
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Conv2d 保留 2D 空间拓扑
        self.keypoint_conv = nn.Conv2d(token_dim, num_keypoints, kernel_size=1)

        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, grid_h),
            torch.linspace(-1, 1, grid_w),
            indexing="ij",
        )
        self.register_buffer(
            "coords", torch.stack([x_coords.flatten(), y_coords.flatten()])
        )

    def forward(self, x):
        # x: [B, num_tokens, token_dim]
        B, N, D = x.shape

        # 还原 2D 拓扑: [B, D, grid_h, grid_w]
        x_2d = x.permute(0, 2, 1).view(B, D, self.grid_h, self.grid_w)

        # Conv2d + softmax: [B, K, grid_h, grid_w] → [B, K, N]
        attention = self.keypoint_conv(x_2d)
        attention = attention.flatten(2)
        attention = F.softmax(attention, dim=-1)

        # 关键点坐标: [B, K, 2]
        keypoints = torch.matmul(attention, self.coords.permute(1, 0))

        # 关键点特征: [B, K, D] (attention 加权平均)
        keypoint_features = torch.matmul(attention, x)

        return keypoints, keypoint_features
