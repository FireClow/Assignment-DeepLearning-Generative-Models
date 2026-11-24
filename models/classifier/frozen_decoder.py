"""Decoder trained on frozen 64-dimensional classifier features.

    64 -> 256 -> 784, then sigmoid and reshape to 1x28x28.

The classifier that produced the features is not part of this module, so its
parameters cannot receive gradients from this reconstruction loss.
"""

from __future__ import annotations

import torch
from torch import nn

IMAGE_SHAPE = (1, 28, 28)
FLAT_DIM = 1 * 28 * 28


class FrozenDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, FLAT_DIM),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"expected [B, {self.latent_dim}], got {tuple(latent.shape)}")
        flat = self.net(latent)
        return flat.reshape(latent.shape[0], *IMAGE_SHAPE)


class FrozenLatentReconstructor(nn.Module):
    """Image-in, image-out wrapper so the stage-2 metric function can score it.

    The classifier stays under ``no_grad``. Reconstruction is the first return
    value, matching ``VanillaAE.forward``.
    """

    def __init__(self, classifier: nn.Module, decoder: FrozenDecoder) -> None:
        super().__init__()
        self.classifier = classifier
        self.decoder = decoder

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            latent = self.classifier.extract_features(images)
        return self.decoder(latent), latent
