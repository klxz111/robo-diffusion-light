import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
except ImportError:
    raise ImportError(
        "diffusers is required for DiffusionActionHead. "
        "Install with: pip install diffusers"
    )


class SinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish"""

    def __init__(self, inp, out, kernel, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp, out, kernel, padding=kernel // 2),
            nn.GroupNorm(n_groups, out),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1d(nn.Module):
    """1D residual block with FiLM conditioning."""

    def __init__(self, in_ch, out_ch, cond_dim, kernel=3, n_groups=8):
        super().__init__()
        self.conv1 = Conv1dBlock(in_ch, out_ch, kernel, n_groups)
        self.conv2 = Conv1dBlock(out_ch, out_ch, kernel, n_groups)
        self.cond_mlp = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_ch * 2))
        self.residual_conv = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, cond):
        out = self.conv1(x)
        cond_embed = self.cond_mlp(cond).unsqueeze(-1)
        scale, bias = cond_embed.chunk(2, dim=1)
        out = scale * out + bias
        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1d(nn.Module):
    """1D temporal U-Net with FiLM conditioning.

    Architecture:
        Encoder: 3 stages of (ResBlock x2 + StridedConv1d)
        Bottleneck: ResBlock x2
        Decoder: 3 stages of (Concat skip + ResBlock x2 + ConvTranspose1d)
        Final: Conv1d to action dimension
    """

    def __init__(
        self,
        action_dim=2,
        horizon=16,
        down_dims=(256, 512, 1024),
        cond_dim=64,
        kernel_size=5,
        n_groups=8,
    ):
        super().__init__()
        self.horizon = horizon
        total_cond = 64 + cond_dim

        # Encoder channel plan
        # Stage 0: action_dim -> down_dims[0]
        # Stage 1: down_dims[0] -> down_dims[1]
        # Stage 2: down_dims[1] -> down_dims[2]
        enc_in = [action_dim] + list(down_dims[:-1])  # [2, 256, 512]
        enc_out = list(down_dims)  # [256, 512, 1024]

        self.down_modules = nn.ModuleList()
        for i in range(len(down_dims)):
            is_last = i == len(down_dims) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(
                            enc_in[i], enc_out[i], total_cond, kernel_size, n_groups
                        ),
                        ConditionalResidualBlock1d(
                            enc_out[i], enc_out[i], total_cond, kernel_size, n_groups
                        ),
                        nn.Conv1d(enc_out[i], enc_out[i], 3, stride=2, padding=1)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        bottleneck_ch = down_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1d(
                    bottleneck_ch, bottleneck_ch, total_cond, kernel_size, n_groups
                ),
                ConditionalResidualBlock1d(
                    bottleneck_ch, bottleneck_ch, total_cond, kernel_size, n_groups
                ),
            ]
        )

        # Decoder channel plan (reversed)
        # Stage 0: bottleneck(1024) + skip[2](1024) -> 1024 -> upsample -> 1024
        # Stage 1: prev_up(1024) + skip[1](512) -> 512 -> upsample -> 512
        # Stage 2: prev_up(512) + skip[0](256) -> 256
        dec_out = list(reversed(down_dims))  # [1024, 512, 256]
        skips = list(
            reversed(down_dims)
        )  # skip channels from encoder: [1024, 512, 256]

        self.up_modules = nn.ModuleList()
        for i in range(len(down_dims)):
            is_last = i == len(down_dims) - 1
            if i == 0:
                dec_in_ch = bottleneck_ch + skips[i]  # 1024 + 1024 = 2048
            else:
                dec_in_ch = dec_out[i - 1] + skips[i]  # 1024+512=1536, 512+256=768

            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(
                            dec_in_ch, dec_out[i], total_cond, kernel_size, n_groups
                        ),
                        ConditionalResidualBlock1d(
                            dec_out[i], dec_out[i], total_cond, kernel_size, n_groups
                        ),
                        nn.ConvTranspose1d(
                            dec_out[i], dec_out[i], 4, stride=2, padding=1
                        )
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(self, x, timestep, global_cond=None):
        x = x.permute(0, 2, 1)
        t_emb = SinusoidalPosEmb(64)(timestep.to(x.device).float())
        cond = t_emb
        if global_cond is not None:
            cond = torch.cat([cond, global_cond], dim=-1)

        skips = []
        for res1, res2, downsample in self.down_modules:
            x = res1(x, cond)
            x = res2(x, cond)
            skips.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, cond)

        for (res1, res2, upsample), skip in zip(self.up_modules, reversed(skips)):
            x = torch.cat([x, skip], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)
            x = upsample(x)

        return self.final_conv(x).permute(0, 2, 1)


class DiffusionActionHead(nn.Module):
    """Diffusion Policy action head.

    Wraps a ConditionalUnet1d + DDPMScheduler for training and inference.

    Training:
        loss = head.compute_loss(global_cond, action)

    Inference:
        action_chunk = head.generate(global_cond, batch_size)
    """

    def __init__(
        self,
        action_dim=2,
        horizon=16,
        n_action_steps=4,
        global_cond_dim=512,
        down_dims=(64, 128, 256),
        num_train_timesteps=100,
        clip_sample_range=1.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_action_steps = n_action_steps

        self.unet = ConditionalUnet1d(
            action_dim=action_dim,
            horizon=horizon,
            down_dims=down_dims,
            cond_dim=global_cond_dim,
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            clip_sample_range=clip_sample_range,
            prediction_type="epsilon",
        )

        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            clip_sample_range=clip_sample_range,
            prediction_type="epsilon",
        )

    def compute_loss(self, global_cond, action):
        """Compute diffusion loss.

        Args:
            global_cond: [B, global_cond_dim]  (Mamba output)
            action:      [B, horizon, action_dim]  (normalized to [-1, 1])

        Returns:
            MSE loss between predicted noise and actual noise
        """
        B = action.shape[0]
        eps = torch.randn_like(action)
        t = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=action.device,
        )
        noisy = self.noise_scheduler.add_noise(action, eps, t)
        pred = self.unet(noisy, t, global_cond)
        return F.mse_loss(pred, eps)

    @torch.no_grad()
    def generate(
        self, global_cond, batch_size=1, num_inference_steps=None, use_ddim=True
    ):
        """Generate action chunk via iterative denoising.

        Args:
            global_cond: [B, global_cond_dim]
            batch_size: number of samples to generate
            num_inference_steps: defaults to 10 for DDIM, 100 for DDPM
            use_ddim: if True, use DDIMScheduler (faster); else DDPMScheduler

        Returns:
            [B, n_action_steps, action_dim]  (still in [-1, 1] space)
        """
        device = global_cond.device
        sample = torch.randn(batch_size, self.horizon, 2, device=device)

        scheduler = self.inference_scheduler if use_ddim else self.noise_scheduler
        if num_inference_steps is None:
            num_inference_steps = (
                10 if use_ddim else scheduler.config.num_train_timesteps
            )
        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred = self.unet(sample, t_batch, global_cond)
            sample = scheduler.step(pred, t, sample).prev_sample

        return sample[:, : self.n_action_steps]

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
