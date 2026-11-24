"""Vanilla GAN and DCGAN."""

from models.gans.dcgan import DCGANDiscriminator, DCGANGenerator
from models.gans.vanilla_gan import VanillaDiscriminator, VanillaGenerator

__all__ = [
    "DCGANDiscriminator",
    "DCGANGenerator",
    "VanillaDiscriminator",
    "VanillaGenerator",
]
