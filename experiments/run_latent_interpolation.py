"""Linear latent interpolation from checkpoints that are already trained.

    .venv/bin/python experiments/run_latent_interpolation.py

Autoencoder endpoints are two test images of different classes, encoded by
each model. VAE uses the posterior mean, not a random sample. GAN endpoints
are two draws from N(0, I) with seed 42, so the path is in noise space rather
than in an encoder latent. No weights are updated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloaders
from data.splits import load_split
from models.autoencoders.sparse_ae import SparseAE
from models.autoencoders.vae import VariationalAE
from models.autoencoders.vanilla_ae import VanillaAE
from models.gans.dcgan import DCGANGenerator
from models.gans.vanilla_gan import VanillaGenerator
from training.autoencoder_trainer import load_checkpoint as load_autoencoder
from training.gan_trainer import gan_to_unit, load_checkpoint as load_gan
from visualization.plot_utils import class_name

ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
FIGURE_PATH = ROOT / "outputs" / "figures" / "figure_04_latent_interpolation.png"
TENSOR_PATH = ROOT / "outputs" / "metrics" / "latent_interpolation.pt"
META_PATH = ROOT / "outputs" / "metrics" / "latent_interpolation.json"
AE_DIR = ROOT / "outputs" / "checkpoints" / "autoencoders"
GAN_DIR = ROOT / "outputs" / "checkpoints" / "gans"
ROW_ORDER = ("vanilla_ae", "sparse_ae", "vae", "vanilla_gan", "dcgan")
ROW_TITLES = {
    "vanilla_ae": "Vanilla AE",
    "sparse_ae": "Sparse AE",
    "vae": "VAE",
    "vanilla_gan": "Vanilla GAN",
    "dcgan": "DCGAN",
}


def interpolate(start: torch.Tensor, end: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    """Return ``[len(alphas), latent]`` from ``(1 - a) * start + a * end``."""
    scale = alphas.to(dtype=start.dtype).view(-1, *([1] * start.ndim))
    return (1.0 - scale) * start + scale * end


def contrasting_pair(images: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """First test image, then the first later image from another class."""
    first = int(labels[0].item())
    for index in range(1, int(labels.shape[0])):
        if int(labels[index].item()) != first:
            return 0, index
    raise RuntimeError("the test batch does not contain two classes")


def encode_mean(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """One latent row. VAE returns mu so the path does not depend on epsilon."""
    if isinstance(model, VariationalAE):
        mu, _logvar = model.encode(image)
        return mu[0]
    return model.encode(image)[0]


def decode_row(model: torch.nn.Module, latent: torch.Tensor, gan: bool) -> torch.Tensor:
    images = model.decode(latent) if not gan else model(latent)
    if gan:
        return gan_to_unit(images)
    return images.clamp(0.0, 1.0)


def save_figure(rows: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(ROW_ORDER), len(ALPHAS), figsize=(12, 9))
    for row_index, name in enumerate(ROW_ORDER):
        for column, alpha in enumerate(ALPHAS):
            axis = axes[row_index, column]
            axis.imshow(
                rows[name][column, 0].detach().cpu().numpy(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(f"α={alpha:.1f}", fontsize=10)
            if column == 0:
                axis.set_ylabel(ROW_TITLES[name], fontsize=9)
    figure.suptitle("Figure 4. Linear latent interpolation", fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with (ROOT / "configs" / "data.yaml").open(encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle)
    split = load_split(ROOT / data_config["split_path"])
    loaders = build_dataloaders(
        ROOT / data_config["data_root"],
        split,
        batch_size=128,
        num_workers=0,
        seed=int(data_config["seed"]),
    )
    images, labels = next(iter(loaders["test"]))
    index_a, index_b = contrasting_pair(images, labels)
    endpoint_images = torch.stack((images[index_a], images[index_b])).to(device)
    pooled_a = int(split["test"][index_a])
    pooled_b = int(split["test"][index_b])
    label_a = int(labels[index_a].item())
    label_b = int(labels[index_b].item())

    autoencoders = {
        "vanilla_ae": VanillaAE(64),
        "sparse_ae": SparseAE(64),
        "vae": VariationalAE(64),
    }
    filenames = {
        "vanilla_ae": "vanilla_ae_best.pt",
        "sparse_ae": "sparse_ae_best.pt",
        "vae": "vae_best.pt",
    }
    for name, model in autoencoders.items():
        load_autoencoder(AE_DIR / filenames[name], model, map_location=device)
        model.to(device)
        model.eval()

    generators = {
        "vanilla_gan": VanillaGenerator(100),
        "dcgan": DCGANGenerator(100),
    }
    gan_files = {"vanilla_gan": "vanilla_gan_final.pt", "dcgan": "dcgan_final.pt"}
    for name, generator in generators.items():
        load_gan(GAN_DIR / gan_files[name], generator, map_location=device)
        generator.to(device)
        generator.eval()

    noise_generator = torch.Generator().manual_seed(42)
    gan_noise = torch.randn(2, 100, generator=noise_generator)
    alphas = torch.tensor(ALPHAS, dtype=torch.float32, device=device)
    rows: dict[str, torch.Tensor] = {}
    latent_rows: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, model in autoencoders.items():
            start = encode_mean(model, endpoint_images[0:1])
            end = encode_mean(model, endpoint_images[1:2])
            path = interpolate(start, end, alphas)
            decoded = decode_row(model, path, gan=False)
            direct_start = decode_row(model, start.unsqueeze(0), gan=False)
            direct_end = decode_row(model, end.unsqueeze(0), gan=False)
            if not torch.allclose(decoded[0], direct_start[0], atol=1e-5):
                raise RuntimeError(f"{name} alpha 0 does not match its endpoint")
            if not torch.allclose(decoded[-1], direct_end[0], atol=1e-5):
                raise RuntimeError(f"{name} alpha 1 does not match its endpoint")
            rows[name] = decoded.cpu()
            latent_rows[name] = path.cpu()
        for name, generator in generators.items():
            start = gan_noise[0].to(device)
            end = gan_noise[1].to(device)
            path = interpolate(start, end, alphas)
            decoded = decode_row(generator, path, gan=True)
            direct_start = decode_row(generator, start.unsqueeze(0), gan=True)
            direct_end = decode_row(generator, end.unsqueeze(0), gan=True)
            if not torch.allclose(decoded[0], direct_start[0], atol=1e-5):
                raise RuntimeError(f"{name} alpha 0 does not match its endpoint")
            if not torch.allclose(decoded[-1], direct_end[0], atol=1e-5):
                raise RuntimeError(f"{name} alpha 1 does not match its endpoint")
            rows[name] = decoded.cpu()
            latent_rows[name] = path.cpu()

    for name, tensor in rows.items():
        if tensor.shape != (len(ALPHAS), 1, 28, 28):
            raise RuntimeError(f"{name} interpolation shape is {tuple(tensor.shape)}")
        if float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0:
            raise RuntimeError(f"{name} pixels leave [0, 1]")
    save_figure(rows, FIGURE_PATH)
    torch.save(
        {
            "alphas": list(ALPHAS),
            "images": torch.stack([rows[name] for name in ROW_ORDER]),
            "latents": {name: latent_rows[name] for name in ROW_ORDER},
            "endpoint_images": endpoint_images.cpu(),
            "gan_noise": gan_noise,
        },
        TENSOR_PATH,
    )
    metadata = {
        "formula": "z(alpha) = (1 - alpha) * z_A + alpha * z_B",
        "alphas": list(ALPHAS),
        "seed": 42,
        "autoencoder_endpoints": {
            "test_position_a": index_a,
            "test_position_b": index_b,
            "pooled_index_a": pooled_a,
            "pooled_index_b": pooled_b,
            "label_a": label_a,
            "label_b": label_b,
            "class_a": class_name(label_a),
            "class_b": class_name(label_b),
            "shared_by": ["vanilla_ae", "sparse_ae", "vae"],
            "vae_latent": "posterior mean mu, not a reparameterized sample",
        },
        "gan_endpoints": {
            "space": "training noise prior N(0, I), dimension 100, not an encoder latent",
            "seed": 42,
            "shared_by": ["vanilla_gan", "dcgan"],
        },
        "pixel_range": "[0, 1]",
        "gan_display": "(tanh + 1) / 2, clamped to [0, 1]",
        "models": list(ROW_ORDER),
        "figure": str(FIGURE_PATH.relative_to(ROOT)),
        "weights_updated": False,
    }
    META_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"endpoints={class_name(label_a)}/{class_name(label_b)} "
        f"positions={index_a},{index_b} figure={FIGURE_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
