"""Fully connected GAN. The discriminator returns logits, not probabilities."""

from __future__ import annotations

import torch
from torch import nn

from models.gans.weights import init_gan_weights

IMAGE_SHAPE = (1, 28, 28)
FLAT_DIM = 1 * 28 * 28


class VanillaGenerator(nn.Module):
    """z [B, 100] -> image [B, 1, 28, 28] in (-1, 1) via Tanh."""

    def __init__(self, latent_dim: int = 100) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, FLAT_DIM),
            nn.Tanh(),
        )
        self.apply(init_gan_weights)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.latent_dim:
            raise ValueError(f"expected noise [B, {self.latent_dim}], got {tuple(noise.shape)}")
        return self.net(noise).view(noise.size(0), *IMAGE_SHAPE)


class VanillaDiscriminator(nn.Module):
    """Image [B, 1, 28, 28] -> logit [B, 1]. No sigmoid on this output."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FLAT_DIM, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        self.apply(init_gan_weights)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != IMAGE_SHAPE:
            raise ValueError(f"expected [B, 1, 28, 28], got {tuple(images.shape)}")
        return self.net(images)
