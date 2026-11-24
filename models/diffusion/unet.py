"""Small U-Net that predicts DDPM noise on a 28x28 grayscale image.

Spatial sizes use kernel 4, stride 2, padding 1, which lands on integers:

    conv:          (28 + 2 - 4) / 2 + 1 = 14, then 14 -> 7
    convtranspose: (7 - 1) * 2 - 2 + 4 = 14, then 14 -> 28

``match_spatial`` center-crops or pads if a future kernel choice misses those
sizes. On this 28x28 schedule it is a no-op.

Channel path:

    input                 [B, 1, 28, 28]
    initial conv          [B, 64, 28, 28]
    residual + skip 28    [B, 64, 28, 28]
    down                  [B, 128, 14, 14]
    residual + skip 14    [B, 128, 14, 14]
    down                  [B, 128, 7, 7]
    bottleneck, two res   [B, 128, 7, 7]
    up                    [B, 128, 14, 14]
    concat skip 14        [B, 256, 14, 14]
    residual              [B, 128, 14, 14]
    up                    [B, 64, 28, 28]
    concat skip 28        [B, 128, 28, 28]
    residual              [B, 64, 28, 28]
    final conv            [B, 1, 28, 28]

The output is epsilon, not a generated image. There is no sigmoid or tanh.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Map each integer timestep to a ``dim``-vector of sin and cos features."""
    if dim % 2 != 0:
        raise ValueError("sinusoidal embedding dimension must be even")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat((angles.sin(), angles.cos()), dim=1)


def group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def match_spatial(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Make ``source`` match the height and width of ``reference``.

    Padding is applied first when the source is smaller. Cropping then removes
    any extra border, split evenly on both sides. Equal sizes return ``source``.
    """
    _, _, height, width = source.shape
    _, _, target_h, target_w = reference.shape
    if height == target_h and width == target_w:
        return source
    pad_h = max(target_h - height, 0)
    pad_w = max(target_w - width, 0)
    if pad_h or pad_w:
        source = F.pad(
            source,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        )
        _, _, height, width = source.shape
    top = max((height - target_h) // 2, 0)
    left = max((width - target_w) // 2, 0)
    return source[:, :, top : top + target_h, left : left + target_w]


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.proj(sinusoidal_embedding(timesteps, self.dim))


class ResidualBlock(nn.Module):
    """Two GroupNorm-SiLU-Conv layers plus a projected time vector."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, images: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(images)))
        shift = self.time_proj(F.silu(time_embedding)).unsqueeze(-1).unsqueeze(-1)
        hidden = hidden + shift
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(images)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.conv(images)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.conv(images)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.time_dim = time_dim
        width = base_channels
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.init_conv = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.enc_28 = ResidualBlock(width, width, time_dim)
        self.down_to_14 = Downsample(width, width * 2)
        self.enc_14 = ResidualBlock(width * 2, width * 2, time_dim)
        self.down_to_7 = Downsample(width * 2, width * 2)
        self.mid_1 = ResidualBlock(width * 2, width * 2, time_dim)
        self.mid_2 = ResidualBlock(width * 2, width * 2, time_dim)
        self.up_to_14 = Upsample(width * 2, width * 2)
        self.dec_14 = ResidualBlock(width * 4, width * 2, time_dim)
        self.up_to_28 = Upsample(width * 2, width)
        self.dec_28 = ResidualBlock(width * 2, width, time_dim)
        self.final_conv = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)

    def forward(self, images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"U-Net input must be [B, C, H, W], got {tuple(images.shape)}")
        if timesteps.shape != (images.shape[0],):
            raise ValueError(
                f"timesteps must have shape {(images.shape[0],)}, got {tuple(timesteps.shape)}"
            )
        time_embedding = self.time_embedding(timesteps)
        hidden = self.init_conv(images)
        skip_28 = self.enc_28(hidden, time_embedding)
        hidden = self.enc_14(self.down_to_14(skip_28), time_embedding)
        skip_14 = hidden
        hidden = self.down_to_7(hidden)
        hidden = self.mid_1(hidden, time_embedding)
        hidden = self.mid_2(hidden, time_embedding)
        hidden = match_spatial(self.up_to_14(hidden), skip_14)
        hidden = self.dec_14(torch.cat((hidden, skip_14), dim=1), time_embedding)
        hidden = match_spatial(self.up_to_28(hidden), skip_28)
        hidden = self.dec_28(torch.cat((hidden, skip_28), dim=1), time_embedding)
        predicted = self.final_conv(hidden)
        expected = (images.shape[0], self.out_channels, images.shape[2], images.shape[3])
        if tuple(predicted.shape) != expected:
            raise RuntimeError(f"U-Net output is {tuple(predicted.shape)}, expected {expected}")
        return predicted
