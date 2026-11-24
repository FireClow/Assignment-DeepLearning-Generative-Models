"""One DDPM chain shown at fixed noise levels, including the final x_0."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from models.diffusion.ddpm import TRAJECTORY_LEVELS

ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = ROOT / "outputs" / "figures" / "ddpm_denoising_trajectory.png"


def save_trajectory(
    frames: dict[int, torch.Tensor],
    path: Path = FIGURE_PATH,
) -> None:
    """``frames`` maps a noise level to a [1, 1, 28, 28] tensor in [0, 1]."""
    missing = [level for level in TRAJECTORY_LEVELS if level not in frames]
    if missing:
        raise ValueError(f"trajectory is missing levels {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(TRAJECTORY_LEVELS), figsize=(16, 2.6))
    for axis, level in zip(axes, TRAJECTORY_LEVELS):
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
        label = "x_T" if level == TRAJECTORY_LEVELS[0] else ("x_0" if level == 0 else f"t={level}")
        axis.set_title(label, fontsize=9)
        axis.axis("off")
    figure.suptitle("DDPM reverse diffusion on one chain", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
