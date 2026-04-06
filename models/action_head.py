import torch
import torch.nn as nn


class ActionHead(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        # 双分支动作头，输出 2D 坐标 (x, y)
        self.x_net = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1), nn.Tanh()
        )
        self.y_net = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1), nn.Tanh()
        )

    def forward(self, x):
        # x: [B, T, 64]  ->  [B, T, 2]  值域 [-1, 1]
        px = self.x_net(x)
        py = self.y_net(x)
        return torch.cat([px, py], dim=-1)
