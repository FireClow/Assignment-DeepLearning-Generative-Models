"""Deterministic 80/10/10 indices over the pooled Fashion-MNIST set.

The official archive is 60_000 train images plus 10_000 test images. This
project pools both, then cuts 56_000 / 7_000 / 7_000 with a private generator
seeded at 42. A private generator keeps the split stable even if other code
calls torch.manual_seed before training.

Pooled index map:
    0 .. 59_999      official FashionMNIST(train=True)
    60_000 .. 69_999 official FashionMNIST(train=False)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

TRAIN_RATIO_NUMERATOR = 8
VAL_RATIO_NUMERATOR = 1
RATIO_DENOMINATOR = 10


def split_counts(n: int) -> tuple[int, int, int]:
    """Return train, validation, and test sizes that sum to n.

    Integer arithmetic is used so 70_000 becomes exactly 56_000 / 7_000 / 7_000.
    The test size absorbs any remainder when n is not divisible by 10.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    n_train = (n * TRAIN_RATIO_NUMERATOR) // RATIO_DENOMINATOR
    n_val = (n * VAL_RATIO_NUMERATOR) // RATIO_DENOMINATOR
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"n={n} is too small for an 80/10/10 split")
    return n_train, n_val, n_test


def make_split_indices(n: int, seed: int = 42) -> dict:
    """Permute 0..n-1 with Generator(seed) and cut the locked ratios.

    The returned lists are JSON-serializable. Two calls with the same n and
    seed return equal lists, regardless of the global PyTorch RNG state.
    """
    n_train, n_val, n_test = split_counts(n)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n, generator=generator)
    train = permutation[:n_train].tolist()
    val = permutation[n_train : n_train + n_val].tolist()
    test = permutation[n_train + n_val :].tolist()
    if not (len(train) == n_train and len(val) == n_val and len(test) == n_test):
        raise RuntimeError("split sizes diverged from split_counts")
    return {
        "dataset": "fashion_mnist",
        "seed": seed,
        "n": n,
        "counts": {"train": n_train, "val": n_val, "test": n_test},
        "index_map": "0..59999 official train, 60000..69999 official test",
        "train": train,
        "val": val,
        "test": test,
    }


def canonical_digest(payload: dict) -> str:
    """SHA-256 of the fields that define the split, for reload checks."""
    body = {
        "seed": payload["seed"],
        "n": payload["n"],
        "train": payload["train"],
        "val": payload["val"],
        "test": payload["test"],
    }
    encoded = json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_split(payload: dict, path: Path) -> None:
    """Write the split JSON. Parent directories are created if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    path.write_text(encoded + "\n", encoding="utf-8")


def load_split(path: Path) -> dict:
    """Load a split JSON produced by save_split."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("seed", "n", "train", "val", "test")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"split file {path} is missing keys: {missing}")
    return payload
