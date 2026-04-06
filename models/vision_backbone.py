import torch
import torch.nn as nn
from torchvision import models


class VisionBackbone(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        res18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 截断到 layer2 输出，保留 12x12 空间特征（128 通道）
        # 96 -> 48(conv1) -> 24(maxpool) -> 24(layer1) -> 12(layer2)
        self.backbone = nn.Sequential(*(list(res18.children())[:-4]))

        for param in self.backbone.parameters():
            param.requires_grad = False

        # 投影: [B, 144, 128] → [B, 144, embed_dim]
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x):
        # 输入 x: [B, 3, 96, 96]
        with torch.no_grad():
            feature = self.backbone(x)  # 输出: [B, 256, 12, 12]

        B, C, H, W = feature.shape
        feature = feature.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, 144, 256]
        out = self.proj(feature)  # [B, 144, 512]
        return out
