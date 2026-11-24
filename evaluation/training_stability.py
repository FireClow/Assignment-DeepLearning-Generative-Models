"""Scalar checks recorded beside the samples. They are warnings, not a collapse verdict."""

from __future__ import annotations

import math

import torch


def stability_flags(
    generator_loss: float,
    discriminator_loss: float,
    real_accuracy: float,
    fake_accuracy: float,
    fake_images: torch.Tensor,
) -> list[str]:
    """Flag unstable signals. Mode collapse still has to be confirmed visually."""
    mean = float(fake_images.detach().mean())
    std = float(fake_images.detach().std())
    values = [generator_loss, discriminator_loss, real_accuracy, fake_accuracy, mean, std]
    if not all(math.isfinite(value) for value in values):
        return ["non_finite"]
    flags: list[str] = []
    if mean > 0.95:
        flags.append("near_white")
    if mean < -0.95:
        flags.append("near_black")
    if std < 0.05:
        flags.append("low_variation")
    if real_accuracy > 0.98 and fake_accuracy > 0.98:
        flags.append("discriminator_separates_easily")
    if generator_loss > 10.0 and discriminator_loss < 0.1:
        flags.append("generator_loss_high")
    return flags
