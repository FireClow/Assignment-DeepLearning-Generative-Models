"""Shared figure helpers. Images are already in [0, 1]."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FASHION_CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def class_name(label: int) -> str:
    return FASHION_CLASSES[int(label)]


def save_gray_grid(images: torch.Tensor, path: Path, title: str) -> None:
    """Save a square grid. `images` is [N, 1, 28, 28] in [0, 1]."""
    count = int(images.shape[0])
    side = int(math.sqrt(count))
    if side * side != count:
        raise ValueError(f"grid needs a square count, got {count}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(side, side, figsize=(side * 1.5, side * 1.5))
    for index, axis in enumerate(axes.ravel()):
        axis.imshow(
            images[index, 0].detach().cpu().numpy(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_training_curves(histories: dict[str, list[dict]], path: Path) -> None:
    """One panel per model.

    Reconstruction and total loss share the left axis. When the penalty term
    lives on a much larger scale, it is drawn on the right axis so it does not
    flatten the reconstruction curve.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = (
        ("vanilla_ae", "Vanilla AE", "penalty"),
        ("sparse_ae", "Sparse AE", "sparsity L1"),
        ("vae", "VAE", "KL"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for axis, (key, title, penalty_name) in zip(axes, panels):
        history = histories[key]
        epochs = [row["epoch"] for row in history]
        lines = []
        lines.append(axis.plot(epochs, [row["train_total"] for row in history], label="train total")[0])
        lines.append(axis.plot(epochs, [row["val_total"] for row in history], label="val total")[0])
        lines.append(
            axis.plot(
                epochs,
                [row["train_reconstruction"] for row in history],
                linestyle="--",
                label="train reconstruction",
            )[0]
        )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Total / reconstruction")
        axis.grid(True, alpha=0.3)
        penalty = [row["train_penalty"] for row in history]
        if key != "vanilla_ae":
            penalty_axis = axis.twinx()
            lines.append(
                penalty_axis.plot(
                    epochs,
                    penalty,
                    color="tab:red",
                    linestyle=":",
                    label=f"train {penalty_name}",
                )[0]
            )
            penalty_axis.set_ylabel(penalty_name)
        labels = [line.get_label() for line in lines]
        axis.legend(lines, labels, fontsize=8, loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
