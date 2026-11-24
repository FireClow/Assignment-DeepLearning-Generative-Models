"""Transpose projections and Gaussian sampling from classifier features.

Both projections follow the required formulas and omit bias:

    z_hat = ReLU(W3^T y)
    h_hat = ReLU(W2^T z_hat)
    x_hat = Sigmoid(W1^T h_hat)

    x_hat = Sigmoid(W1^T ReLU(W2^T z_MLP))

``W1``, ``W2``, and ``W3`` use PyTorch layout ``[out, in]``. For a batch of
row vectors the transposes become right-multiplies: ``y @ W3``, ``z @ W2``,
and ``h @ W1``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.classifier.mlp_classifier import FLAT_DIM


def top_down_transpose(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
) -> torch.Tensor:
    """Map the ten one-hot class vectors to images of shape ``[10, 1, 28, 28]``."""
    if tuple(w3.shape) != (10, 64) or tuple(w2.shape) != (64, 256) or tuple(w1.shape) != (256, 784):
        raise ValueError(f"unexpected weight shapes {tuple(w1.shape)}, {tuple(w2.shape)}, {tuple(w3.shape)}")
    one_hot = torch.eye(w3.shape[0], device=w3.device, dtype=w3.dtype)
    latent = F.relu(one_hot @ w3)
    hidden = F.relu(latent @ w2)
    pixels = torch.sigmoid(hidden @ w1)
    return pixels.reshape(one_hot.shape[0], 1, 28, 28)


def latent_transpose(latent: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Project ``z_MLP`` back to pixels. Output shape is ``[B, 1, 28, 28]``."""
    if latent.ndim != 2 or latent.shape[1] != w2.shape[0]:
        raise ValueError(f"latent shape {tuple(latent.shape)} does not match W2 {tuple(w2.shape)}")
    hidden = F.relu(latent @ w2)
    pixels = torch.sigmoid(hidden @ w1)
    if pixels.shape[1] != FLAT_DIM:
        raise ValueError(f"projected pixels have width {pixels.shape[1]}, expected {FLAT_DIM}")
    return pixels.reshape(latent.shape[0], 1, 28, 28)


def latent_statistics(features: torch.Tensor, std_epsilon: float) -> dict[str, torch.Tensor | int | float]:
    """Mean and per-dimension standard deviation of ``z_MLP``.

    ``torch.std`` uses the unbiased sample estimator. A dimension whose
    standard deviation is below ``std_epsilon`` is raised to that floor so the
    Gaussian scale stays finite. The count of such dimensions is recorded.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be [N, D], got {tuple(features.shape)}")
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=True)
    clamped = int((std < std_epsilon).sum().item())
    return {
        "mean": mean,
        "std": std.clamp_min(std_epsilon),
        "raw_std": std,
        "clamped_dimensions": clamped,
        "std_epsilon": float(std_epsilon),
        "count": int(features.shape[0]),
    }


def sample_gaussian_prior(
    mean: torch.Tensor,
    std: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw ``z = mu + epsilon * sigma`` with ``epsilon ~ N(0, I)``."""
    epsilon = torch.randn(count, mean.shape[0], generator=generator, dtype=mean.dtype)
    return mean.cpu() + epsilon * std.cpu()


def classification_scores(
    predicted: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict[str, float | list[float]]:
    """Accuracy plus macro precision, recall, and F1 from integer predictions."""
    if predicted.shape != labels.shape:
        raise ValueError("predictions and labels must share one shape")
    per_precision: list[float] = []
    per_recall: list[float] = []
    per_f1: list[float] = []
    for class_index in range(num_classes):
        true_positive = int(((predicted == class_index) & (labels == class_index)).sum().item())
        predicted_positive = int((predicted == class_index).sum().item())
        actual_positive = int((labels == class_index).sum().item())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_precision.append(precision)
        per_recall.append(recall)
        per_f1.append(f1)
    accuracy = float((predicted == labels).float().mean().item()) if labels.numel() else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": sum(per_precision) / num_classes,
        "macro_recall": sum(per_recall) / num_classes,
        "macro_f1": sum(per_f1) / num_classes,
        "per_class_precision": per_precision,
        "per_class_recall": per_recall,
        "per_class_f1": per_f1,
    }
