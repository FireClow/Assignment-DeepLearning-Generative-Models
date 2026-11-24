"""Variational autoencoder with a diagonal Gaussian approximate posterior.

Encoder backbone: 784 -> 256 -> 128. Two linear heads then produce
mu and logvar, each in R^64. A sample is drawn with the reparameterization
trick so the sampling step still has a gradient:

    z = mu + exp(0.5 * logvar) * epsilon,   epsilon ~ N(0, I)

Decoder: 64 -> 128 -> 256 -> 784, sigmoid, reshape to 1x28x28.

The objective is the standard ELBO, with beta = 1:

    total = BCE(x_hat, x) + beta * KL(q(z|x) || N(0, I))
    KL = -0.5 * mean(sum(1 + logvar - mu^2 - exp(logvar), dim=1))

BCE is summed over the 784 pixels and then averaged over the batch. That
matches the KL term, which is summed over the 64 latent dimensions and then
averaged over the batch. logvar is clamped only to keep exp(logvar) finite.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.autoencoders.vanilla_ae import FLAT_DIM, IMAGE_SHAPE, _check_image

LOGVAR_MIN = -20.0
LOGVAR_MAX = 20.0


class VariationalAE(nn.Module):
    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FLAT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, FLAT_DIM),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mu and logvar, each shaped [B, latent_dim]."""
        _check_image(images)
        hidden = self.backbone(images)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Draw z = mu + sigma * epsilon without blocking the gradient."""
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        flat = self.decoder(latent)
        return flat.view(latent.size(0), *IMAGE_SHAPE)

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(images)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent), mu, logvar, latent


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form KL from a diagonal Gaussian to N(0, I). Scalar batch mean."""
    per_image = torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return -0.5 * per_image.mean()


def vae_loss(
    reconstruction: torch.Tensor,
    images: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, per-image-sum BCE, and the KL term."""
    per_pixel = F.binary_cross_entropy(reconstruction, images, reduction="none")
    reconstruction_loss = per_pixel.flatten(1).sum(dim=1).mean()
    kl_loss = kl_divergence(mu, logvar)
    total = reconstruction_loss + beta * kl_loss
    return total, reconstruction_loss, kl_loss
