"""Shared GAN weight initialization.

DCGAN conv layers follow the DCGAN paper: Normal(0, 0.02) weights.
BatchNorm scale is Normal(1, 0.02) and its bias is zero. Vanilla GAN has no
convolutions, so the same rule is applied to its Linear layers.
"""

from __future__ import annotations

from torch import nn


def init_gan_weights(module: nn.Module) -> None:
    name = module.__class__.__name__
    if "Conv" in name or "Linear" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.zeros_(module.bias.data)
