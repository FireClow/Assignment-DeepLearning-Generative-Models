"""Stage-5 figures. Every image is grayscale in [0, 1] with the same display scale."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from visualization.plot_utils import class_name

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "outputs" / "figures"


def _show(axis: plt.Axes, image: torch.Tensor) -> None:
    axis.imshow(image[0].detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])


def save_weight_transpose(images: torch.Tensor, path: Path | None = None) -> None:
    path = path or FIGURE_DIR / "classifier_weight_transpose.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 10, figsize=(15, 2.4))
    for index, axis in enumerate(axes):
        _show(axis, images[index])
        axis.set_title(class_name(index), fontsize=8)
    figure.suptitle("Top-down weight transpose, one template per class")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_latent_transpose(
    originals: torch.Tensor,
    projections: torch.Tensor,
    labels: torch.Tensor,
    path: Path | None = None,
) -> None:
    path = path or FIGURE_DIR / "classifier_latent_transpose.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(originals.shape[0])
    figure, axes = plt.subplots(2, count, figsize=(1.6 * count, 3.8))
    for index in range(count):
        _show(axes[0, index], originals[index])
        _show(axes[1, index], projections[index])
        axes[0, index].set_title(class_name(int(labels[index])), fontsize=8)
    axes[0, 0].set_ylabel("Original", fontsize=8)
    axes[1, 0].set_ylabel("Transpose", fontsize=8)
    figure.suptitle("Latent transpose projection of the same test images")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_frozen_comparison(
    originals: torch.Tensor,
    autoencoder: torch.Tensor,
    decoder: torch.Tensor,
    labels: torch.Tensor,
    path: Path | None = None,
) -> None:
    path = path or FIGURE_DIR / "frozen_latent_decoder_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(originals.shape[0])
    rows = (originals, autoencoder, decoder)
    names = ("Original", "Vanilla AE", "MLP frozen decoder")
    figure, axes = plt.subplots(3, count, figsize=(1.6 * count, 5.2))
    for row, (images, name) in enumerate(zip(rows, names)):
        for index in range(count):
            _show(axes[row, index], images[index])
            if row == 0:
                axes[row, index].set_title(class_name(int(labels[index])), fontsize=8)
        axes[row, 0].set_ylabel(name, fontsize=8)
    figure.suptitle("Same test images, Vanilla AE versus the frozen MLP decoder")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_prior_samples(images: torch.Tensor, path: Path | None = None) -> None:
    path = path or FIGURE_DIR / "mlp_random_prior_sampling.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(images.shape[0])
    side = 4 if count == 16 else int(count**0.5)
    figure, axes = plt.subplots(side, side, figsize=(8, 8))
    for index, axis in enumerate(axes.ravel()):
        _show(axis, images[index])
    figure.suptitle("Samples from N(mean, variance) of test z_MLP, decoded")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _add_row(figure: plt.Figure, spec, images: torch.Tensor, titles: list[str], ylabel: str) -> None:
    inner = spec.subgridspec(1, int(images.shape[0]), wspace=0.05)
    for index in range(int(images.shape[0])):
        axis = figure.add_subplot(inner[index])
        _show(axis, images[index])
        if titles[index]:
            axis.set_title(titles[index], fontsize=7)
        if index == 0:
            axis.set_ylabel(ylabel, fontsize=8)


def save_figure_03(
    originals: torch.Tensor,
    templates: torch.Tensor,
    projections: torch.Tensor,
    autoencoder: torch.Tensor,
    reconstructions: torch.Tensor,
    prior: torch.Tensor,
    labels: torch.Tensor,
    path: Path | None = None,
) -> None:
    """Six panels. The image rows share the same eight test images."""
    path = path or FIGURE_DIR / "figure_03_classifier_proof.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(14, 15))
    outer = figure.add_gridspec(6, 1, height_ratios=[1.0, 1.15, 1.0, 1.0, 1.0, 2.2], hspace=0.48)
    _add_row(figure, outer[0], originals, [class_name(int(label)) for label in labels], "Original")
    _add_row(
        figure,
        outer[1],
        templates,
        [class_name(index) for index in range(int(templates.shape[0]))],
        "Top-down",
    )
    blank = [""] * int(projections.shape[0])
    _add_row(figure, outer[2], projections, blank, "Latent\ntranspose")
    _add_row(figure, outer[3], autoencoder, blank, "Vanilla AE")
    _add_row(figure, outer[4], reconstructions, blank, "Frozen\ndecoder")
    prior_grid = outer[5].subgridspec(2, 8, wspace=0.05, hspace=0.12)
    for index in range(min(16, int(prior.shape[0]))):
        axis = figure.add_subplot(prior_grid[index // 8, index % 8])
        _show(axis, prior[index])
        if index == 0:
            axis.set_ylabel("Prior", fontsize=8)
    figure.suptitle(
        "Figure 3. Original, transpose, Vanilla AE, frozen decoder, and prior samples",
        fontsize=12,
    )
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
