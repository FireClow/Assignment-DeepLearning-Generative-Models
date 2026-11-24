"""DDPM noise-prediction loss and ancestral reverse sampling.

Training draws a random timestep and scores the U-Net with mean squared error
against the Gaussian noise that created ``x_t``. That quantity is a noise
prediction loss, not a reconstruction loss.

Sampling starts at ``x_T ~ N(0, I)`` and walks t = T-1, ..., 0. Posterior
noise is added when t > 0. The t = 0 step returns the reverse mean and does
not add noise; that tensor is the estimate of ``x_0``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.diffusion.noise_scheduler import NoiseScheduler
from models.diffusion.unet import UNet

TRAJECTORY_LEVELS = (1000, 875, 750, 625, 500, 375, 250, 125, 0)


def unit_to_diffusion(images: torch.Tensor) -> torch.Tensor:
    """Map dataset pixels [0, 1] to the diffusion range [-1, 1]. Apply once."""
    return images * 2.0 - 1.0


def diffusion_to_unit(images: torch.Tensor) -> torch.Tensor:
    """Map a reverse-process tensor to [0, 1] for figures. Values are clamped."""
    return ((images + 1.0) * 0.5).clamp(0.0, 1.0)


class DDPM(nn.Module):
    def __init__(self, unet: UNet, scheduler: NoiseScheduler) -> None:
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler

    def noise_prediction_loss(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean squared error between the U-Net output and the injected epsilon."""
        if timesteps is None:
            timesteps = self.scheduler.sample_timesteps(x0.shape[0], x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        noisy = self.scheduler.add_noise(x0, timesteps, noise)
        predicted = self.unet(noisy, timesteps)
        return F.mse_loss(predicted, noise)

    def reverse_step(self, x_t: torch.Tensor, t_index: int) -> torch.Tensor:
        """One ancestral step from noise level ``t_index`` toward ``t_index - 1``."""
        if t_index < 0 or t_index >= self.scheduler.num_timesteps:
            raise ValueError(f"timestep {t_index} is outside 0..{self.scheduler.num_timesteps - 1}")
        timesteps = torch.full(
            (x_t.shape[0],),
            t_index,
            device=x_t.device,
            dtype=torch.long,
        )
        predicted = self.unet(x_t, timesteps)
        mean = self.scheduler.reverse_mean(x_t, timesteps, predicted)
        if t_index == 0:
            return mean
        variance = self.scheduler.extract(self.scheduler.posterior_variance, timesteps, x_t)
        return mean + torch.sqrt(variance) * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        frames: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run all T reverse steps. Optional ``frames`` stores one shared chain.

        A stored level is the tensor *before* the reverse step that leaves that
        level, except level 0, which is the finished ``x_0``. Level ``T`` is the
        initial ``x_T``. Every stored frame comes from this same loop.
        """
        was_training = self.unet.training
        self.unet.eval()
        try:
            total = self.scheduler.num_timesteps
            noise = torch.randn((batch_size, self.unet.in_channels, 28, 28), generator=generator)
            current = noise.to(device=device, dtype=torch.float32)
            if frames is not None:
                frames[total] = current.detach().cpu()
            for t_index in range(total - 1, -1, -1):
                level = t_index + 1
                if frames is not None and level != total and level in TRAJECTORY_LEVELS:
                    frames[level] = current.detach().cpu()
                current = self.reverse_step(current, t_index)
            if frames is not None:
                frames[0] = current.detach().cpu()
            if not torch.isfinite(current).all():
                raise RuntimeError("reverse diffusion produced a non-finite sample")
            return current
        finally:
            self.unet.train(was_training)
