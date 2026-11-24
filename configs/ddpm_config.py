"""Training settings for the Fashion-MNIST DDPM.

The split, seed, and batch size are read from configs/data.yaml. This file
does not change that contract. Diffusion-specific values live here so the
schedule and the U-Net width can be inspected without editing the data file.

``grad_clip`` is None. Gradient clipping is only applied if this value is set,
and a run that sets it must say so in the metrics file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DDPMConfig:
    seed: int = 42
    batch_size: int = 128
    learning_rate: float = 2e-4
    max_epochs: int = 30
    patience: int = 5
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 64
    time_dim: int = 256
    grad_clip: float | None = None
    num_workers: int = 2
    data_root: str = "data/raw"
    split_path: str = "data/splits/fashion_mnist_seed42.json"
    validation_seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def load_ddpm_config(root: Path) -> DDPMConfig:
    """Read the locked data fields, then keep the stage-4 diffusion defaults."""
    with (root / "configs" / "data.yaml").open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return DDPMConfig(
        seed=int(data["seed"]),
        batch_size=int(data["batch_size"]),
        num_workers=int(data["num_workers"]),
        data_root=str(data["data_root"]),
        split_path=str(data["split_path"]),
        validation_seed=int(data["seed"]),
    )
