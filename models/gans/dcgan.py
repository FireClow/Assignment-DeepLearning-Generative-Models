"""Convolutional GAN sized for one 28x28 grayscale channel.

Generator feature maps, with output = (input - 1) * stride - 2 * padding + kernel:
    z is [B, 100, 1, 1]
    ConvTranspose 7x7, stride 1 -> 7x7
    ConvTranspose 4x4, stride 2, padding 1 -> 14x14
    ConvTranspose 4x4, stride 2, padding 1 -> 28x28

Discriminator:
    Conv 4x4, stride 2, padding 1: 28 -> 14
    Conv 4x4, stride 2, padding 1: 14 -> 7
    Flatten 128 * 7 * 7, which is measured with a probe tensor at init.
"""

from __future__ import annotations

import torch
from torch import nn

from models.gans.weights import init_gan_weights


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self.apply(init_gan_weights)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        if noise.ndim == 2:
            noise = noise.unsqueeze(-1).unsqueeze(-1)
        if noise.ndim != 4 or noise.shape[1:] != (self.latent_dim, 1, 1):
            raise ValueError(
                f"expected noise [B, {self.latent_dim}, 1, 1], got {tuple(noise.shape)}"
            )
        images = self.net(noise)
        if tuple(images.shape[1:]) != (1, 28, 28):
            raise RuntimeError(f"DCGAN generator produced {tuple(images.shape)}, expected [B, 1, 28, 28]")
        return images


class DCGANDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        with torch.no_grad():
            probe = self.features(torch.zeros(1, 1, 28, 28))
        self.feature_shape = tuple(int(value) for value in probe.shape[1:])
        self.feature_dim = int(probe.numel())
        self.classifier = nn.Linear(self.feature_dim, 1)
        self.apply(init_gan_weights)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError(f"expected [B, 1, 28, 28], got {tuple(images.shape)}")
        features = self.features(images)
        if tuple(features.shape[1:]) != self.feature_shape:
            raise RuntimeError(
                f"DCGAN feature map is {tuple(features.shape)}, expected [B, {self.feature_shape}]"
            )
        return self.classifier(features.flatten(1))
