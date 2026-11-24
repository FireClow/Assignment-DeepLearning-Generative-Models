"""Loss curves for every trained family, plus one DDPM denoising chain.

Each panel has its own y-axis. Autoencoder reconstruction, VAE ELBO, GAN
adversarial loss, and DDPM noise MSE are not the same quantity.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from models.diffusion.ddpm import TRAJECTORY_LEVELS

ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = ROOT / "outputs" / "figures" / "figure_05_training_curves.png"
LOG_DIR = ROOT / "outputs" / "logs"


def save_figure_05(
    frames: dict[int, torch.Tensor],
    path: Path = FIGURE_PATH,
    log_dir: Path = LOG_DIR,
) -> None:
    histories = {name: json.loads((log_dir / f"{name}_history.json").read_text(encoding="utf-8")) for name in (
        "vanilla_ae",
        "sparse_ae",
        "vae",
        "vanilla_gan",
        "dcgan",
        "ddpm",
    )}
    missing = [level for level in TRAJECTORY_LEVELS if level not in frames]
    if missing:
        raise ValueError(f"trajectory is missing levels {missing}")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(14.5, 8.2))
    grid = figure.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.35)
    axes = [figure.add_subplot(grid[0, index]) for index in range(4)]

    vanilla = histories["vanilla_ae"]
    sparse = histories["sparse_ae"]
    epochs = [row["epoch"] for row in vanilla]
    axes[0].plot(epochs, [row["train_reconstruction"] for row in vanilla], label="vanilla train")
    axes[0].plot(epochs, [row["val_reconstruction"] for row in vanilla], label="vanilla val")
    axes[0].plot(
        [row["epoch"] for row in sparse],
        [row["train_reconstruction"] for row in sparse],
        label="sparse train recon",
    )
    axes[0].plot(
        [row["epoch"] for row in sparse],
        [row["train_total"] for row in sparse],
        linestyle="--",
        label="sparse train total",
    )
    axes[0].set_title("AE reconstruction MSE")
    axes[0].set_ylabel("MSE")

    vae = histories["vae"]
    axes[1].plot([row["epoch"] for row in vae], [row["train_total"] for row in vae], label="train ELBO")
    axes[1].plot([row["epoch"] for row in vae], [row["val_total"] for row in vae], label="val ELBO")
    axes[1].set_title("VAE ELBO")
    axes[1].set_ylabel("BCE + KL")

    for name, label in (("vanilla_gan", "Vanilla"), ("dcgan", "DCGAN")):
        history = histories[name]
        gan_epochs = [row["epoch"] for row in history]
        axes[2].plot(gan_epochs, [row["generator_loss"] for row in history], label=f"{label} G")
        axes[2].plot(
            gan_epochs,
            [row["discriminator_loss"] for row in history],
            linestyle="--",
            label=f"{label} D",
        )
    axes[2].set_title("GAN adversarial loss")
    axes[2].set_ylabel("BCE-with-logits")

    ddpm = histories["ddpm"]
    axes[3].plot([row["epoch"] for row in ddpm], [row["train_noise_loss"] for row in ddpm], label="train")
    axes[3].plot([row["epoch"] for row in ddpm], [row["val_noise_loss"] for row in ddpm], label="val")
    axes[3].set_title("DDPM noise prediction MSE")
    axes[3].set_ylabel("MSE(epsilon)")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7, loc="best")

    inner = grid[1, :].subgridspec(1, len(TRAJECTORY_LEVELS), wspace=0.08)
    for index, level in enumerate(TRAJECTORY_LEVELS):
        axis = figure.add_subplot(inner[index])
        image = frames[level]
        if image.ndim == 4:
            image = image[0]
        axis.imshow(
            image[0].detach().cpu().numpy(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        label = "x_T" if level == TRAJECTORY_LEVELS[0] else ("x_0" if level == 0 else str(level))
        axis.set_title(label, fontsize=8)
        axis.axis("off")

    figure.suptitle("Figure 5. Training curves and one DDPM reverse chain", fontsize=13)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
