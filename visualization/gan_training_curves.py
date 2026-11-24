"""GAN loss curves and the fixed-noise progress sheet."""

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

CURVE_PATH = ROOT / "outputs" / "figures" / "gan_training_curves.png"
PROGRESS_PATH = ROOT / "outputs" / "figures" / "gan_training_progress.png"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "gans"
PROGRESS_EPOCHS = (1, 25, 50)


def save_gan_curves(histories: dict[str, list[dict]], path: Path = CURVE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    styles = (("vanilla_gan", "Vanilla GAN"), ("dcgan", "DCGAN"))
    for name, label in styles:
        history = histories[name]
        epochs = [row["epoch"] for row in history]
        axes[0, 0].plot(epochs, [row["generator_loss"] for row in history], label=label)
        axes[0, 1].plot(epochs, [row["discriminator_loss"] for row in history], label=label)
        axes[1, 0].plot(epochs, [row["discriminator_real_accuracy"] for row in history], label=f"{label} real")
        axes[1, 0].plot(
            epochs,
            [row["discriminator_fake_accuracy"] for row in history],
            linestyle="--",
            label=f"{label} fake",
        )
        axes[1, 1].plot(epochs, [row["seconds"] for row in history], label=label)
    axes[0, 0].set_title("Generator loss")
    axes[0, 1].set_title("Discriminator loss")
    axes[1, 0].set_title("Discriminator accuracy")
    axes[1, 1].set_title("Seconds per epoch")
    for axis in axes.ravel():
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_progress_figure(
    snapshots: dict[str, dict[int, torch.Tensor]],
    path: Path = PROGRESS_PATH,
) -> None:
    """snapshots[model][epoch] is [8, 1, 28, 28] in [0, 1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    models = ("vanilla_gan", "dcgan")
    titles = {"vanilla_gan": "Vanilla GAN", "dcgan": "DCGAN"}
    figure, axes = plt.subplots(len(models) * len(PROGRESS_EPOCHS), 8, figsize=(12, 10))
    row = 0
    for name in models:
        for epoch in PROGRESS_EPOCHS:
            images = snapshots[name][epoch]
            for column in range(8):
                axis = axes[row, column]
                axis.imshow(
                    images[column, 0].detach().cpu().numpy(),
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
                axis.set_xticks([])
                axis.set_yticks([])
                if column == 0:
                    axis.set_ylabel(f"{titles[name]}\nepoch {epoch}", fontsize=8)
            row += 1
    figure.suptitle("Fixed-noise samples at the start, middle, and end", fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def snapshots_from_checkpoints(device: torch.device) -> dict[str, dict[int, torch.Tensor]]:
    snapshots: dict[str, dict[int, torch.Tensor]] = {}
    builders = {"vanilla_gan": VanillaGenerator, "dcgan": DCGANGenerator}
    for name, builder in builders.items():
        snapshots[name] = {}
        for epoch in PROGRESS_EPOCHS:
            generator = builder()
            payload = load_checkpoint(
                CHECKPOINT_DIR / f"{name}_epoch_{epoch:03d}.pt",
                generator,
                map_location=device,
            )
            generator.to(device)
            images = generate_images(generator, payload["fixed_noise"][:8], device)
            snapshots[name][epoch] = images.cpu()
    return snapshots
