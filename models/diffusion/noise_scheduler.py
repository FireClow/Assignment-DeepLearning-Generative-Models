"""Linear DDPM noise schedule and the closed-form forward process.

Timesteps are integers in ``0 .. T-1``. Index 0 is the least noisy step.
The state before the first reverse step is ``x_T``, which is not itself an index.

Forward process:

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

The reverse mean used at sampling time is the epsilon parameterization from
Ho et al.:

    mu = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * epsilon)

Posterior variance follows the forward-process posterior, with
``alpha_bar_{-1} = 1`` so the variance at t = 0 is zero.
"""

from __future__ import annotations

import torch
from torch import nn


class NoiseScheduler(nn.Module):
    """Deterministic buffers. Nothing here is learned."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ) -> None:
        super().__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        if schedule != "linear":
            raise ValueError(f"only a linear beta schedule is implemented, got {schedule}")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("beta schedule must satisfy 0 < beta_start < beta_end < 1")

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat((torch.ones(1, dtype=torch.float32), alpha_bar[:-1]))
        posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_mean_coef_x0 = betas * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_mean_coef_xt = torch.sqrt(alphas) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

        self.num_timesteps = int(timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.schedule = schedule
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_mean_coef_x0", posterior_mean_coef_x0)
        self.register_buffer("posterior_mean_coef_xt", posterior_mean_coef_xt)

    def extract(self, values: torch.Tensor, timesteps: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Gather one schedule value per batch row and reshape it for broadcasting."""
        if timesteps.ndim != 1 or timesteps.shape[0] != like.shape[0]:
            raise ValueError(
                f"timesteps must be rank-1 with length {like.shape[0]}, got {tuple(timesteps.shape)}"
            )
        selected = values.gather(0, timesteps.to(device=values.device, dtype=torch.long))
        view = [like.shape[0]] + [1] * (like.ndim - 1)
        return selected.reshape(view).to(dtype=like.dtype)

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Uniform integer timestep for each image, in ``0 .. T-1``."""
        return torch.randint(
            0,
            self.num_timesteps,
            (batch_size,),
            device=device,
            generator=generator,
        )

    def add_noise(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Build ``x_t`` and return it. ``noise`` is the training target epsilon."""
        if noise.shape != x0.shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} does not match {tuple(x0.shape)}")
        signal = self.extract(self.sqrt_alpha_bar, timesteps, x0)
        residual = self.extract(self.sqrt_one_minus_alpha_bar, timesteps, x0)
        return signal * x0 + residual * noise

    def predict_x0(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Invert the forward equation for the clean image implied by epsilon."""
        signal = self.extract(self.sqrt_alpha_bar, timesteps, x_t)
        residual = self.extract(self.sqrt_one_minus_alpha_bar, timesteps, x_t)
        return (x_t - residual * epsilon) / signal

    def posterior_mean(
        self,
        x_t: torch.Tensor,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Mean of ``q(x_{t-1} | x_t, x_0)`` from the two stored coefficients."""
        coef_x0 = self.extract(self.posterior_mean_coef_x0, timesteps, x_t)
        coef_xt = self.extract(self.posterior_mean_coef_xt, timesteps, x_t)
        return coef_x0 * x0 + coef_xt * x_t

    def reverse_mean(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Epsilon-parameterized reverse mean. Equivalent to ``posterior_mean``."""
        beta = self.extract(self.betas, timesteps, x_t)
        alpha = self.extract(self.alphas, timesteps, x_t)
        residual = self.extract(self.sqrt_one_minus_alpha_bar, timesteps, x_t)
        return (1.0 / torch.sqrt(alpha)) * (x_t - beta / residual * epsilon)
