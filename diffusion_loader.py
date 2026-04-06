"""
V4 DataLoader for Diffusion Policy training.

方案 A: 预加载所有帧到 RAM（消除视频解码瓶颈）
方案 B: 增强移到 GPU（消除 CPU-GPU 传输瓶颈）
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.expanduser("~/.cache/huggingface/datasets")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "lerobot/src")))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

SAFE_DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "data/raw_pusht")
)
os.makedirs(SAFE_DATA_ROOT, exist_ok=True)

HORIZON = 16
N_OBS_STEPS = 4

# Action normalization params
ACT_MIN = torch.tensor([12.0, 25.0])
ACT_MAX = torch.tensor([511.0, 511.0])


def _preload_dataset(root):
    """方案 A: 一次性加载所有帧到 RAM，消除视频解码开销。"""
    print("  Preloading all frames to RAM...")
    t0 = time.time()

    dataset = LeRobotDataset(
        "lerobot/pusht",
        root=root,
        video_backend="pyav",
    )

    total_frames = len(dataset)
    # 预分配 numpy 数组
    images = np.zeros((total_frames, 96, 96, 3), dtype=np.uint8)
    actions = np.zeros((total_frames, 2), dtype=np.float32)

    for i in range(total_frames):
        item = dataset[i]
        img = item["observation.image"]  # [96, 96, 3] uint8
        act = item["action"]  # [2] float32
        if isinstance(img, torch.Tensor):
            images[i] = img.permute(1, 2, 0).numpy()  # [C,H,W] → [H,W,C]
        else:
            images[i] = np.array(img)
        if isinstance(act, torch.Tensor):
            actions[i] = act.numpy()
        else:
            actions[i] = np.array(act)

        if (i + 1) % 5000 == 0:
            print(
                "    Loaded %d/%d frames (%.1fs)"
                % (i + 1, total_frames, time.time() - t0)
            )

    elapsed = time.time() - t0
    print(
        "  Preload complete: %d frames, %.1f MB RAM, %.1fs"
        % (total_frames, images.nbytes / 1024 / 1024, elapsed)
    )

    # 构建时序样本索引
    fps = dataset.fps
    obs_timestamps = [-(N_OBS_STEPS - 1 - i) / fps for i in range(N_OBS_STEPS)]
    action_timestamps = [i / fps for i in range(HORIZON)]

    # 找到每个样本的有效起始索引
    valid_indices = []
    for i in range(N_OBS_STEPS - 1, total_frames - HORIZON + 1):
        valid_indices.append(i)

    return images, actions, valid_indices, fps


class PreloadedPushTDataset(Dataset):
    """预加载的 PushT 数据集，零磁盘 I/O。"""

    def __init__(self, images, actions, valid_indices, fps):
        self.images = images  # [N, 96, 96, 3] uint8
        self.actions = actions  # [N, 2] float32
        self.valid_indices = valid_indices
        self.fps = fps

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        center = self.valid_indices[idx]

        # 观测帧：过去 N_OBS_STEPS 帧
        obs_frames = []
        for t_offset in range(-(N_OBS_STEPS - 1), 1):
            frame_idx = center + t_offset
            if frame_idx < 0:
                frame_idx = 0
            img = self.images[frame_idx].astype(np.float32) / 255.0  # [96, 96, 3]
            obs_frames.append(img)
        obs_seq = np.stack(obs_frames, axis=0)  # [N_OBS_STEPS, 96, 96, 3]

        # 动作：未来 HORIZON 步
        action_start = center
        action_end = center + HORIZON
        if action_end <= len(self.actions):
            action_seq = self.actions[action_start:action_end]  # [HORIZON, 2]
        else:
            action_seq = self.actions[action_start:]
            pad_len = HORIZON - len(action_seq)
            if pad_len > 0:
                action_seq = np.concatenate(
                    [action_seq, np.repeat(action_seq[-1:], pad_len, axis=0)], axis=0
                )

        # 转换为 [C, H, W] 格式
        obs_seq = np.transpose(obs_seq, (0, 3, 1, 2))  # [N_OBS_STEPS, 3, 96, 96]

        return {
            "observation.image": torch.from_numpy(obs_seq).float(),
            "action": torch.from_numpy(action_seq).float(),
        }


def augment_batch_gpu(
    batch, max_trans_px=15, max_rot_deg=10, action_noise_std=3.0, augment_prob=0.8
):
    """方案 B: GPU 端数据增强（消除 CPU-GPU 传输瓶颈）。

    在训练循环中调用，batch 已在 GPU 上。
    """
    if torch.rand(1, device=batch["observation.image"].device).item() >= augment_prob:
        return batch

    images = batch["observation.image"]  # [B, n_obs, C, H, W]
    actions = batch["action"]  # [B, horizon, 2]

    B, n_obs, C, H, W = images.shape
    N = B * n_obs

    images = images.view(N, C, H, W)

    # 1. 随机平移
    tx = (torch.rand(N, device=images.device) * 2 - 1) * max_trans_px
    ty = (torch.rand(N, device=images.device) * 2 - 1) * max_trans_px
    tx_norm = (tx / W).view(N, 1, 1)
    ty_norm = (ty / H).view(N, 1, 1)
    grid_x = (
        torch.linspace(-1, 1, W, device=images.device).view(1, 1, -1).expand(N, H, -1)
        + tx_norm
    )
    grid_y = (
        torch.linspace(-1, 1, H, device=images.device).view(1, -1, 1).expand(N, -1, W)
        + ty_norm
    )
    grid = torch.stack([grid_x, grid_y], dim=-1)
    images = F.grid_sample(images, grid, align_corners=False, padding_mode="zeros")

    tx_batch = tx.view(B, n_obs).mean(dim=1)
    ty_batch = ty.view(B, n_obs).mean(dim=1)
    actions = actions.clone()
    actions[:, :, 0] += tx_batch.unsqueeze(1)
    actions[:, :, 1] += ty_batch.unsqueeze(1)

    # 2. 随机旋转
    theta_deg = (torch.rand(N, device=images.device) * 2 - 1) * max_rot_deg
    theta_rad = torch.deg2rad(theta_deg)
    cos_a = torch.cos(theta_rad)
    sin_a = torch.sin(theta_rad)

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=images.device),
        torch.linspace(-1, 1, W, device=images.device),
        indexing="ij",
    )
    xx = xx.view(1, H, W).expand(N, -1, -1)
    yy = yy.view(1, H, W).expand(N, -1, -1)
    cos_ = cos_a.view(N, 1, 1)
    sin_ = sin_a.view(N, 1, 1)
    grid_rot_x = cos_ * xx - sin_ * yy
    grid_rot_y = sin_ * xx + cos_ * yy
    grid_rot = torch.stack([grid_rot_x, grid_rot_y], dim=-1)
    images = F.grid_sample(images, grid_rot, align_corners=False, padding_mode="zeros")

    center_x, center_y = 48.0, 48.0
    cos_batch = cos_a.view(B, n_obs).mean(dim=1).unsqueeze(1)
    sin_batch = sin_a.view(B, n_obs).mean(dim=1).unsqueeze(1)
    ax = actions[:, :, 0] - center_x
    ay = actions[:, :, 1] - center_y
    actions[:, :, 0] = cos_batch * ax - sin_batch * ay + center_x
    actions[:, :, 1] = sin_batch * ax + cos_batch * ay + center_y

    # 3. 颜色抖动
    bf = 1.0 + (torch.rand(N, 1, 1, 1, device=images.device) * 2 - 1) * 0.2
    images = images * bf
    cf = 1.0 + (torch.rand(N, 1, 1, 1, device=images.device) * 2 - 1) * 0.2
    mean_c = images.mean(dim=(2, 3), keepdim=True)
    images = (images - mean_c) * cf + mean_c
    sf = 1.0 + (torch.rand(N, 1, 1, 1, device=images.device) * 2 - 1) * 0.2
    gray = images.mean(dim=1, keepdim=True)
    images = gray + sf * (images - gray)
    images = images.clamp(0, 1)

    # 4. 动作噪声
    actions += torch.randn_like(actions) * action_noise_std

    images = images.view(B, n_obs, C, H, W)

    return {"observation.image": images, "action": actions}


def get_diffusion_loader(batch_size=32, num_workers=4):
    """DataLoader for V4 Diffusion Policy training.

    方案 A: 预加载所有帧到 RAM
    方案 B: 增强移到 GPU（在训练循环中调用 augment_batch_gpu）
    """
    print("Loading lerobot/pusht for V4.2 Diffusion training...")

    images, actions, valid_indices, fps = _preload_dataset(SAFE_DATA_ROOT)

    dataset = PreloadedPushTDataset(images, actions, valid_indices, fps)

    print(
        "V4.2 Diffusion dataset ready: images=[B, %d, C, H, W], actions=[B, %d, 2]"
        % (N_OBS_STEPS, HORIZON)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader
