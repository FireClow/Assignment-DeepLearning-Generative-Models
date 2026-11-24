"""Fréchet distance on Gaussian features.

Image FID is computed only when the official InceptionV3 weights are already
on disk. This module does not download them and does not substitute
discriminator activations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm

INCEPTION_FILENAME = "inception_v3_google-0cc3c7bd.pth"


def frechet_distance(
    mean_real: np.ndarray,
    covariance_real: np.ndarray,
    mean_fake: np.ndarray,
    covariance_fake: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """FID = ||mu_r - mu_f||^2 + Tr(C_r + C_f - 2 * sqrt(C_r C_f))."""
    mean_real = np.atleast_1d(np.asarray(mean_real, dtype=np.float64))
    mean_fake = np.atleast_1d(np.asarray(mean_fake, dtype=np.float64))
    covariance_real = np.atleast_2d(np.asarray(covariance_real, dtype=np.float64))
    covariance_fake = np.atleast_2d(np.asarray(covariance_fake, dtype=np.float64))
    difference = mean_real - mean_fake
    product = covariance_real.dot(covariance_fake)
    covmean = sqrtm(product)
    if not np.isfinite(covmean).all():
        offset = np.eye(covariance_real.shape[0], dtype=np.float64) * eps
        covmean = sqrtm((covariance_real + offset).dot(covariance_fake + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise RuntimeError("matrix square root has a large imaginary part")
        covmean = covmean.real
    value = float(
        difference.dot(difference)
        + np.trace(covariance_real)
        + np.trace(covariance_fake)
        - 2.0 * np.trace(covmean)
    )
    if not np.isfinite(value):
        raise RuntimeError("Fréchet distance is not finite")
    return value


def pretrained_inception_status() -> dict:
    """Look for the torchvision InceptionV3 file. Do not download it."""
    path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / INCEPTION_FILENAME
    if not path.is_file():
        return {
            "available": False,
            "weights_file": INCEPTION_FILENAME,
            "reason": (
                "InceptionV3 ImageNet weights are not in the local torch hub cache. "
                "No download was started, so no FID score was computed. "
                "Discriminator features were not used as a substitute."
            ),
            "protocol_if_weights_exist": (
                "Use the same number of Fashion-MNIST test images and generated images, "
                "map Tanh outputs to [0, 1], repeat grayscale to 3 channels, resize to "
                "299x299, apply ImageNet normalization, and compare Inception pool features "
                "with the same extractor for Vanilla GAN and DCGAN."
            ),
        }
    return {"available": True, "weights_file": str(path)}
