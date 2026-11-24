"""Vanilla autoencoder: a deterministic map x -> z -> x_hat.

Encoder: 784 -> 256 -> 128 -> 64
Decoder: 64 -> 128 -> 256 -> 784, then sigmoid so pixels stay in (0, 1).
The latent vector has no activation. Reconstruction uses mean squared error,
which matches pixels stored in [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

IMAGE_SHAPE = (1, 28, 28)
FLAT_DIM = 1 * 28 * 28


def _check_image(images: torch.Tensor) -> None:
    if images.ndim != 4 or tuple(images.shape[1:]) != IMAGE_SHAPE:
        raise ValueError(f"expected [B, 1, 28, 28], got {tuple(images.shape)}")


class VanillaAE(nn.Module):
    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FLAT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, FLAT_DIM),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return z with shape [B, latent_dim]."""
        _check_image(images)
        return self.encoder(images)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Map z back to pixels. Output shape is [B, 1, 28, 28] in (0, 1)."""
        flat = self.decoder(latent)
        return flat.view(latent.size(0), *IMAGE_SHAPE)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(images)
        return self.decode(latent), latent


def mse_reconstruction(reconstruction: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """Mean squared error over every pixel in the batch. Shape: scalar."""
    return F.mse_loss(reconstruction, images)
