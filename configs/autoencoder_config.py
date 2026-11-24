"""Training settings for Vanilla AE, Sparse AE, and VAE.

The data split itself stays in configs/data.yaml. This file only adds the
optimization choices for stage 2. Test images are never read here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AutoencoderConfig:
    seed: int = 42
    batch_size: int = 128
    latent_dim: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 30
    patience: int = 5
    sparse_lambda: float = 1e-3
    vae_beta: float = 1.0
    num_workers: int = 2
    data_root: str = "data/raw"
    split_path: str = "data/splits/fashion_mnist_seed42.json"

    def to_dict(self) -> dict:
        return asdict(self)


def load_autoencoder_config(root: Path) -> AutoencoderConfig:
    """Start from the locked data contract, then apply the stage-2 defaults."""
    with (root / "configs" / "data.yaml").open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return AutoencoderConfig(
        seed=int(data["seed"]),
        batch_size=int(data["batch_size"]),
        latent_dim=int(data["latent_dim"]),
        sparse_lambda=float(data["sparse_lambda"]),
        num_workers=int(data["num_workers"]),
        data_root=str(data["data_root"]),
        split_path=str(data["split_path"]),
    )
