"""Figure 1: the same eight test images, reconstructed by each autoencoder.

Run after training:

    .venv/bin/python visualization/figure_01_reconstruction.py
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

from configs.autoencoder_config import load_autoencoder_config
from data.dataset import build_dataloaders
from data.splits import load_split
from models.autoencoders.sparse_ae import SparseAE
from models.autoencoders.vae import VariationalAE
from models.autoencoders.vanilla_ae import VanillaAE
from training.autoencoder_trainer import load_checkpoint, resolve_device
from visualization.plot_utils import class_name

FIGURE_PATH = ROOT / "outputs" / "figures" / "figure_01_reconstruction.png"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "autoencoders"
NUM_IMAGES = 8


def first_test_images(num_images: int = NUM_IMAGES) -> tuple[torch.Tensor, torch.Tensor]:
    config = load_autoencoder_config(ROOT)
    split = load_split(ROOT / config.split_path)
    loaders = build_dataloaders(
        ROOT / config.data_root,
        split,
        batch_size=max(config.batch_size, num_images),
        num_workers=0,
        seed=config.seed,
    )
    images, labels = next(iter(loaders["test"]))
    return images[:num_images], labels[:num_images]


def load_trained(model: torch.nn.Module, filename: str, device: torch.device) -> torch.nn.Module:
    load_checkpoint(CHECKPOINT_DIR / filename, model, map_location=device)
    model.to(device)
    model.eval()
    return model


def save_reconstruction_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    reconstructions: dict[str, torch.Tensor],
    path: Path = FIGURE_PATH,
) -> None:
    """Rows are original, Vanilla AE, Sparse AE, and VAE. Columns share one test image."""
    row_names = ("Original", "Vanilla AE", "Sparse AE", "VAE")
    row_tensors = (
        images,
        reconstructions["vanilla_ae"],
        reconstructions["sparse_ae"],
        reconstructions["vae"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, images.shape[0], figsize=(images.shape[0] * 1.7, 8))
    for row, (name, tensor) in enumerate(zip(row_names, row_tensors)):
        for column in range(images.shape[0]):
            axis = axes[row, column]
            axis.imshow(tensor[column, 0].detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(class_name(int(labels[column])), fontsize=9)
            if column == 0:
                axis.set_ylabel(name, fontsize=9)
    figure.suptitle("Figure 1. Test-set reconstructions", fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    device = resolve_device()
    images, labels = first_test_images()
    images = images.to(device)
    models = {
        "vanilla_ae": load_trained(VanillaAE(), "vanilla_ae_best.pt", device),
        "sparse_ae": load_trained(SparseAE(), "sparse_ae_best.pt", device),
        "vae": load_trained(VariationalAE(), "vae_best.pt", device),
    }
    reconstructions = {}
    with torch.no_grad():
        for name, model in models.items():
            reconstructions[name] = model(images)[0].clamp(0.0, 1.0)
    save_reconstruction_grid(images, labels, reconstructions)
    print(f"figure={FIGURE_PATH}", flush=True)


if __name__ == "__main__":
    main()
