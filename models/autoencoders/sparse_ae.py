"""Sparse autoencoder: the vanilla MLP plus an L1 penalty on z.

The encoder matches Vanilla AE, with one added ReLU after the 64-dimensional
layer. ReLU makes every latent unit non-negative, so the L1 term can push
unused units exactly to zero:

    sparsity_loss = mean(|z|)
    total = MSE(x_hat, x) + lambda_sparse * sparsity_loss

lambda_sparse is 0.001 unless the caller overrides it. The three terms are
returned separately so the effect of the penalty can be plotted on its own.
"""

from __future__ import annotations

import torch
from torch import nn

from models.autoencoders.vanilla_ae import (
    FLAT_DIM,
    IMAGE_SHAPE,
    _check_image,
    mse_reconstruction,
)


class SparseAE(nn.Module):
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
            nn.ReLU(),
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
        """Return z >= 0 with shape [B, latent_dim]."""
        _check_image(images)
        return self.encoder(images)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        flat = self.decoder(latent)
        return flat.view(latent.size(0), *IMAGE_SHAPE)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(images)
        return self.decode(latent), latent


def sparse_ae_loss(
    reconstruction: torch.Tensor,
    images: torch.Tensor,
    latent: torch.Tensor,
    lambda_sparse: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, reconstruction MSE, and the unweighted L1 mean."""
    reconstruction_loss = mse_reconstruction(reconstruction, images)
    sparsity_loss = latent.abs().mean()
    total = reconstruction_loss + lambda_sparse * sparsity_loss
    return total, reconstruction_loss, sparsity_loss
