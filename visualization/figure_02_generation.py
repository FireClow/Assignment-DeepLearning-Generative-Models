"""Figure 2 sample grids.

``save_figure_02`` keeps the two-panel GAN figure. ``save_figure_02_comparison``
draws VAE, Vanilla GAN, DCGAN, and DDPM, sixteen images each, on one scale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.gans.dcgan import DCGANGenerator
from models.gans.vanilla_gan import VanillaGenerator
from training.gan_trainer import generate_images, load_checkpoint

FIGURE_PATH = ROOT / "outputs" / "figures" / "figure_02_generation.png"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "gans"


def save_figure_02_comparison(
    vae_images: torch.Tensor,
    vanilla_images: torch.Tensor,
    dcgan_images: torch.Tensor,
    ddpm_images: torch.Tensor,
    path: Path = FIGURE_PATH,
) -> None:
    """Four 4x4 panels. Every tensor is [N, 1, 28, 28] in [0, 1]."""
    panels = (
        (vae_images, "VAE"),
        (vanilla_images, "Vanilla GAN"),
        (dcgan_images, "DCGAN"),
        (ddpm_images, "DDPM"),
    )
    for images, title in panels:
        if images.shape[0] < 16:
            raise ValueError(f"{title} needs 16 images")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(16, 5.8))
    outer = figure.add_gridspec(1, 4, wspace=0.06)
    for column, (images, title) in enumerate(panels):
        inner = outer[column].subgridspec(4, 4, wspace=0.04, hspace=0.04)
        for index in range(16):
            axis = figure.add_subplot(inner[index // 4, index % 4])
            axis.imshow(
                images[index, 0].detach().cpu().numpy(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            axis.set_xticks([])
            axis.set_yticks([])
        figure.text((column + 0.5) / 4.0, 0.98, title, ha="center", va="top", fontsize=13)
    figure.suptitle("Figure 2. Random samples from each generative model", y=0.02, fontsize=11)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_figure_02(
    vanilla_images: torch.Tensor,
    dcgan_images: torch.Tensor,
    path: Path = FIGURE_PATH,
) -> None:
    if vanilla_images.shape[0] < 16 or dcgan_images.shape[0] < 16:
        raise ValueError("each panel needs 16 images")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(12, 7.2))
    outer = figure.add_gridspec(1, 2, wspace=0.08)
    panels = ((vanilla_images[:16], "Vanilla GAN"), (dcgan_images[:16], "DCGAN"))
    for column, (images, title) in enumerate(panels):
        inner = outer[column].subgridspec(4, 4, wspace=0.04, hspace=0.04)
        for index in range(16):
            axis = figure.add_subplot(inner[index // 4, index % 4])
            axis.imshow(
                images[index, 0].detach().cpu().numpy(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            axis.set_xticks([])
            axis.set_yticks([])
        figure.text(0.27 + column * 0.5, 0.98, title, ha="center", va="top", fontsize=14)
    figure.suptitle("Figure 2. Random samples from the fixed noise", y=0.02, fontsize=11)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def load_generator(name: str, device: torch.device) -> tuple[torch.nn.Module, torch.Tensor]:
    if name == "vanilla_gan":
        generator = VanillaGenerator()
        filename = "vanilla_gan_final.pt"
    elif name == "dcgan":
        generator = DCGANGenerator()
        filename = "dcgan_final.pt"
    else:
        raise ValueError(name)
    payload = load_checkpoint(CHECKPOINT_DIR / filename, generator, map_location=device)
    generator.to(device)
    generator.eval()
    return generator, payload["fixed_noise"]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vanilla, vanilla_noise = load_generator("vanilla_gan", device)
    dcgan, dcgan_noise = load_generator("dcgan", device)
    vanilla_images = generate_images(vanilla, vanilla_noise[:16], device)
    dcgan_images = generate_images(dcgan, dcgan_noise[:16], device)
    save_figure_02(vanilla_images, dcgan_images)
    print(f"figure={FIGURE_PATH}", flush=True)


if __name__ == "__main__":
    main()
