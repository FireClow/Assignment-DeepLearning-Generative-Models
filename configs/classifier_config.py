"""Settings for the MLP classifier and the frozen latent decoder.

Seed, batch size, workers, and paths come from configs/data.yaml. This module
does not edit that file. Transpose experiments use the classifier weights
without bias; ``std_epsilon`` is added only to a latent standard deviation
that would otherwise be zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ClassifierConfig:
    seed: int = 42
    batch_size: int = 128
    learning_rate: float = 1e-3
    max_epochs: int = 50
    patience: int = 7
    num_classes: int = 10
    latent_dim: int = 64
    decoder_learning_rate: float = 1e-3
    decoder_max_epochs: int = 50
    decoder_patience: int = 7
    prior_samples: int = 16
    std_epsilon: float = 1e-6
    num_workers: int = 2
    data_root: str = "data/raw"
    split_path: str = "data/splits/fashion_mnist_seed42.json"

    def to_dict(self) -> dict:
        return asdict(self)


def load_classifier_config(root: Path) -> ClassifierConfig:
    with (root / "configs" / "data.yaml").open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return ClassifierConfig(
        seed=int(data["seed"]),
        batch_size=int(data["batch_size"]),
        num_workers=int(data["num_workers"]),
        data_root=str(data["data_root"]),
        split_path=str(data["split_path"]),
    )
