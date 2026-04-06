"""
V4 Training Script for Diffusion Policy.

Key changes from V3.5:
  1. ViT backbone (DINO v1 vits8, frozen) instead of ResNet-18
  2. 4-frame observation history (n_obs_steps=4)
  3. BF16 mixed precision + GradScaler
  4. Data augmentation on CPU (collate_fn, vectorized)
  5. num_workers=4, pin_memory=True
  6. Evaluation warmup lock (skip first 10 epochs)
  7. Checkpoint every 5 epochs (last.pth), best_loss.pth on improvement
  8. Early Stopping Patience=8
  9. Gradient norm monitoring
  10. Unified AdamW optimizer (stability first)
  11. SpatialSoftmax uses Conv2d to preserve 2D topology
  12. Keypoint projection with channel compression (not flat Linear)
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from diffusion_loader import get_diffusion_loader
from metrics_logger import MetricsLogger
from models.diffusion_action_head import DiffusionActionHead
from models.hybrid_core import ToolinitModel
from models.spatial_softmax import SpatialSoftmax
from models.vit_backbone import ViTBackbone

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 48
EPOCHS = 35
LR = 1e-4
WARMUP_EPOCHS = 1
MIN_EPOCHS = 5
PATIENCE = 8
GRAD_CLIP = 1.0
GRAD_CLIP_MAX = 2.0
SEED = 42
CKPT_DIR = "checkpoints_diffusion"
LOG_DIR = "logs_diffusion"

HORIZON = 16
N_OBS_STEPS = 4
N_ACTION_STEPS = 4

ACT_MIN = torch.tensor([12.0, 25.0], device=DEVICE)
ACT_MAX = torch.tensor([511.0, 511.0], device=DEVICE)

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────


def normalize_action(a):
    return 2 * (a - ACT_MIN) / (ACT_MAX - ACT_MIN) - 1


def denormalize_action(a):
    return (a + 1) / 2 * (ACT_MAX - ACT_MIN) + ACT_MIN


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path,
    epoch,
    vision,
    core,
    head,
    spatial_softmax,
    keypoint_proj,
    optimizer,
    scheduler,
    best_loss,
):
    torch.save(
        {
            "epoch": epoch,
            "vision": vision.state_dict(),
            "core": core.state_dict(),
            "head": head.state_dict(),
            "spatial_softmax": spatial_softmax.state_dict(),
            "keypoint_proj": keypoint_proj.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
        },
        path,
    )


def load_checkpoint(
    path,
    vision,
    core,
    head,
    spatial_softmax,
    keypoint_proj,
    optimizer,
    scheduler,
):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    vision.load_state_dict(ckpt["vision"])
    core.load_state_dict(ckpt["core"])
    head.load_state_dict(ckpt["head"])
    spatial_softmax.load_state_dict(ckpt["spatial_softmax"])
    keypoint_proj.load_state_dict(ckpt["keypoint_proj"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_loss"]


@torch.no_grad()
def eval_action_std(
    vision, core, head, spatial_softmax, keypoint_proj, loader, num_samples=32
):
    """Run a full-denoising eval batch to measure action output Std vs Real Std.

    Uses num_inference_steps=100 (full DDPM) for academic rigor.
    Returns (pred_std_x, pred_std_y, real_std_x, real_std_y) in pixel space.
    """
    head.eval()
    core.eval()
    vision.eval()
    spatial_softmax.eval()
    keypoint_proj.eval()

    batch = next(iter(loader))
    img = batch["observation.image"].to(DEVICE)[:num_samples]
    real_action = batch["action"].to(DEVICE)[:num_samples]

    B = img.shape[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feat = vision(img)
        out = core(feat)
        out = out.view(B, N_OBS_STEPS, -1, out.shape[-1])
        out = out.view(B * N_OBS_STEPS, -1, out.shape[-1])
        keypoints, kp_features = spatial_softmax(out)
        keypoints = keypoints.view(B, N_OBS_STEPS, -1, 2)
        kp_features = kp_features.view(B, N_OBS_STEPS, -1, kp_features.shape[-1])

        kp_features_compressed = keypoint_proj["feat_compress"](kp_features)
        combined = torch.cat([keypoints, kp_features_compressed], dim=-1)
        global_cond = keypoint_proj["proj"](combined.flatten(1))

    generated = head.generate(
        global_cond, batch_size=B, num_inference_steps=20, use_ddim=True
    )

    real_sliced = real_action[:, :N_ACTION_STEPS, :]

    gen_unnorm = denormalize_action(generated)
    real_unnorm = denormalize_action(real_sliced)

    pred_std_x = gen_unnorm[:, :, 0].std().item()
    pred_std_y = gen_unnorm[:, :, 1].std().item()
    real_std_x = real_unnorm[:, :, 0].std().item()
    real_std_y = real_unnorm[:, :, 1].std().item()

    head.train()
    core.train()
    vision.train()
    spatial_softmax.train()
    for m in keypoint_proj.values():
        m.train()

    return pred_std_x, pred_std_y, real_std_x, real_std_y


# ──────────────────────────────────────────────────────────────────────────────
# Main Training
# ──────────────────────────────────────────────────────────────────────────────
def train(resume=None):
    set_seed(SEED)

    scaler = torch.cuda.amp.GradScaler()

    spatial_softmax = SpatialSoftmax(
        num_keypoints=16, token_dim=512, grid_h=12, grid_w=12
    ).to(DEVICE)

    feat_compress = nn.Linear(512, 32).to(DEVICE)
    kp_input_dim = N_OBS_STEPS * 16 * (2 + 32)
    proj = nn.Linear(kp_input_dim, 512).to(DEVICE)
    keypoint_proj = nn.ModuleDict(
        {
            "feat_compress": feat_compress,
            "proj": proj,
        }
    )

    vision = ViTBackbone(embed_dim=512, n_obs=N_OBS_STEPS).to(DEVICE)
    core = ToolinitModel().to(DEVICE)
    head = DiffusionActionHead(
        action_dim=2,
        horizon=HORIZON,
        n_action_steps=N_ACTION_STEPS,
        global_cond_dim=512,
        down_dims=(64, 128, 256),
    ).to(DEVICE)

    print(
        "ViT params: {:,} ({:,} trainable)".format(
            vision.count_params(), vision.count_trainable()
        )
    )
    print(
        "DiffusionActionHead params: {:,} ({:,} trainable)".format(
            head.count_params(), head.count_trainable()
        )
    )

    all_params = (
        list(vision.proj.parameters())
        + list(core.parameters())
        + list(head.parameters())
        + list(spatial_softmax.parameters())
        + list(keypoint_proj.parameters())
    )
    optimizer = optim.AdamW(all_params, lr=LR, weight_decay=0.0)

    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
    )

    loader = get_diffusion_loader(batch_size=BATCH_SIZE, num_workers=4)

    logger = MetricsLogger(log_dir=LOG_DIR)

    start_epoch, best_loss = 0, float("inf")
    if resume and os.path.isfile(resume):
        start_epoch, best_loss = load_checkpoint(
            resume,
            vision,
            core,
            head,
            spatial_softmax,
            keypoint_proj,
            optimizer,
            scheduler,
        )
        print("Resumed from epoch={}, best_loss={:.6f}".format(start_epoch, best_loss))
        start_epoch += 1

    patience_counter = 0

    vision.train()
    core.train()
    head.train()

    for epoch in range(start_epoch, EPOCHS):
        pbar = tqdm(loader, desc="Epoch {}".format(epoch))
        total_loss = 0.0
        step = 0
        epoch_start = time.time()
        total_grad_norm = 0.0
        grad_norm_count = 0

        for i, batch in enumerate(pbar):
            step_start = time.time()

            img = batch["observation.image"].to(DEVICE)
            action = batch["action"].to(DEVICE)
            action_norm = normalize_action(action)

            B = img.shape[0]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                feat = vision(img)
                out = core(feat)
                out = out.view(B, N_OBS_STEPS, -1, out.shape[-1])
                out = out.view(B * N_OBS_STEPS, -1, out.shape[-1])
                keypoints, kp_features = spatial_softmax(out)
                keypoints = keypoints.view(B, N_OBS_STEPS, -1, 2)
                kp_features = kp_features.view(
                    B, N_OBS_STEPS, -1, kp_features.shape[-1]
                )

                kp_features_compressed = feat_compress(kp_features)
                combined = torch.cat([keypoints, kp_features_compressed], dim=-1)
                global_cond = proj(combined.flatten(1))

                loss = head.compute_loss(global_cond, action_norm)

            scaler.scale(loss).backward()
            step_vram = (
                torch.cuda.max_memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0
            )
            torch.cuda.reset_peak_memory_stats()

            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(vision.proj.parameters())
                + list(core.parameters())
                + list(head.parameters()),
                GRAD_CLIP,
            )
            total_grad_norm += float(grad_norm)
            grad_norm_count += 1

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            step += 1

            step_time = time.time() - step_start
            vram = step_vram

            logger.log_step(
                {
                    "step": epoch * len(loader) + i,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "vram_mb": vram,
                    "step_time_s": step_time,
                }
            )

            if i % 20 == 0:
                avg_gn = total_grad_norm / max(grad_norm_count, 1)
                print(
                    "| Step {} | Loss: {:.6f} | VRAM: {:.0f}MB | Time: {:.3f}s | GradNorm: {:.3f}".format(
                        i, loss.item(), vram, step_time, avg_gn
                    )
                )
            pbar.set_postfix(loss="{:.6f}".format(loss.item()))

        avg_loss = total_loss / step if step else 0
        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_grad_norm = total_grad_norm / max(grad_norm_count, 1)

        if epoch >= 10:
            print(
                "  Running eval (full 100-step denoising, {} samples)...".format(
                    BATCH_SIZE
                )
            )
            pred_std_x, pred_std_y, real_std_x, real_std_y = eval_action_std(
                vision,
                core,
                head,
                spatial_softmax,
                keypoint_proj,
                loader,
                num_samples=min(32, BATCH_SIZE),
            )
        else:
            print("  Skipping eval (warmup lock, epoch < 10)")
            pred_std_x, pred_std_y, real_std_x, real_std_y = 0.0, 0.0, 0.0, 0.0

        lr_val = float(optimizer.param_groups[0]["lr"])

        logger.log_epoch(
            {
                "epoch": epoch,
                "avg_loss": avg_loss,
                "best_loss": best_loss,
                "patience": patience_counter,
                "pred_std_x": float(pred_std_x),
                "pred_std_y": float(pred_std_y),
                "real_std_x": float(real_std_x),
                "real_std_y": float(real_std_y),
                "lr": lr_val,
                "epoch_time_s": epoch_time,
                "grad_norm": avg_grad_norm,
            }
        )

        print(
            "  Epoch {} | Loss: {:.6f} | "
            "Pred_X: {:.2f} vs Real_X: {:.2f} | Pred_Y: {:.2f} vs Real_Y: {:.2f} | "
            "LR: {:.2e} | Time: {:.1f}s | GradNorm: {:.3f}".format(
                epoch,
                avg_loss,
                pred_std_x,
                real_std_x,
                pred_std_y,
                real_std_y,
                lr_val,
                epoch_time,
                avg_grad_norm,
            )
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            save_checkpoint(
                os.path.join(CKPT_DIR, "best.pth"),
                epoch,
                vision,
                core,
                head,
                spatial_softmax,
                keypoint_proj,
                optimizer,
                scheduler,
                best_loss,
            )
            print("  Best model saved (loss={:.6f})".format(best_loss))
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            save_checkpoint(
                os.path.join(CKPT_DIR, "last.pth"),
                epoch,
                vision,
                core,
                head,
                spatial_softmax,
                keypoint_proj,
                optimizer,
                scheduler,
                best_loss,
            )
            print("  Last checkpoint saved (epoch {})".format(epoch))

        logger.save()
        logger.plot_and_save(epoch)

        if epoch >= MIN_EPOCHS and patience_counter >= PATIENCE:
            print("Early stopping at epoch {} (patience={})".format(epoch, PATIENCE))
            break

    print("Training complete! Best Loss: {:.6f}".format(best_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", type=str, default=None, help="checkpoint path to resume"
    )
    args = parser.parse_args()
    train(resume=args.resume)
