"""Stage 3 settings. The image split still comes from configs/data.yaml."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GanConfig:
    seed: int = 42
    batch_size: int = 128
    latent_dim: int = 100
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    max_epochs: int = 50
    num_workers: int = 2
    data_root: str = "data/raw"
    split_path: str = "data/splits/fashion_mnist_seed42.json"
    fixed_noise_count: int = 64
    grid_images: int = 16

    def to_dict(self) -> dict:
        return asdict(self)


def load_gan_config(root: Path) -> GanConfig:
    with (root / "configs" / "data.yaml").open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return GanConfig(
        seed=int(data["seed"]),
        batch_size=int(data["batch_size"]),
        latent_dim=int(data["noise_dim"]),
        num_workers=int(data["num_workers"]),
        data_root=str(data["data_root"]),
        split_path=str(data["split_path"]),
    )
