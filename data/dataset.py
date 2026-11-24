"""Fashion-MNIST pooled into one dataset, then viewed through a fixed split.

Each item is a pair (image, label):
    image: float32 tensor, shape (1, 28, 28), values in [0, 1]
    label: Python int in 0..9, collated by DataLoader to int64 shape (batch,)

Models that need [-1, 1] (GAN / DCGAN with Tanh) rescale inside their own
training step. This module keeps one shared pixel range so every experiment
sees the same tensors.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import FashionMNIST

OFFICIAL_TRAIN_SIZE = 60_000
OFFICIAL_TEST_SIZE = 10_000
POOLED_SIZE = OFFICIAL_TRAIN_SIZE + OFFICIAL_TEST_SIZE

TO_UNIT_INTERVAL = transforms.ToTensor()


def download_fashion_mnist(root: str | Path, attempts: int = 3) -> None:
    """Download the official train and test archives under root/FashionMNIST.

    The Fashion-MNIST mirror sometimes resets the connection after a file has
    already reached disk. Retrying is safe: torchvision skips any archive whose
    checksum already matches.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            FashionMNIST(root=root, train=True, download=True)
            FashionMNIST(root=root, train=False, download=True)
            return
        except (OSError, RuntimeError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(2)
    raise RuntimeError(f"Fashion-MNIST download failed after {attempts} attempts") from last_error


class PooledFashionMNIST(Dataset):
    """70_000 images addressed by a single pooled index.

    Indices 0..59_999 read the official training archive.
    Indices 60_000..69_999 read the official test archive.
    """

    def __init__(self, root: str | Path) -> None:
        root = Path(root)
        self.train_set = FashionMNIST(
            root=root,
            train=True,
            download=False,
            transform=TO_UNIT_INTERVAL,
        )
        self.test_set = FashionMNIST(
            root=root,
            train=False,
            download=False,
            transform=TO_UNIT_INTERVAL,
        )
        if len(self.train_set) != OFFICIAL_TRAIN_SIZE:
            raise RuntimeError(
                f"official train size is {len(self.train_set)}, expected {OFFICIAL_TRAIN_SIZE}"
            )
        if len(self.test_set) != OFFICIAL_TEST_SIZE:
            raise RuntimeError(
                f"official test size is {len(self.test_set)}, expected {OFFICIAL_TEST_SIZE}"
            )

    def __len__(self) -> int:
        return POOLED_SIZE

    def __getitem__(self, pooled_index: int) -> tuple[torch.Tensor, int]:
        if pooled_index < 0 or pooled_index >= POOLED_SIZE:
            raise IndexError(f"pooled index {pooled_index} outside 0..{POOLED_SIZE - 1}")
        if pooled_index < OFFICIAL_TRAIN_SIZE:
            image, label = self.train_set[pooled_index]
        else:
            image, label = self.test_set[pooled_index - OFFICIAL_TRAIN_SIZE]
        return image, int(label)


def build_dataloaders(
    root: str | Path,
    split: dict,
    batch_size: int = 128,
    num_workers: int = 2,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Build train, val, and test loaders that share one pooled dataset.

    Train is shuffled with a generator seeded by `seed`, so a new process that
    uses the same seed walks the training batches in the same order. Validation
    and test are never shuffled. The underlying split indices still come only
    from `split`, not from this shuffle.
    """
    if split["n"] != POOLED_SIZE:
        raise ValueError(f"split n is {split['n']}, dataset length is {POOLED_SIZE}")

    pooled = PooledFashionMNIST(root)
    subsets = {
        name: Subset(pooled, split[name])
        for name in ("train", "val", "test")
    }
    train_generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    return {
        "train": DataLoader(
            subsets["train"],
            shuffle=True,
            generator=train_generator,
            **loader_kwargs,
        ),
        "val": DataLoader(subsets["val"], shuffle=False, **loader_kwargs),
        "test": DataLoader(subsets["test"], shuffle=False, **loader_kwargs),
    }
