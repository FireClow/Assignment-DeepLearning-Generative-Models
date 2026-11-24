"""Properties of the shared 80/10/10 split. These tests do not download data."""

from __future__ import annotations

import torch

from data.splits import (
    canonical_digest,
    load_split,
    make_split_indices,
    save_split,
    split_counts,
)


def test_fashion_mnist_counts() -> None:
    assert split_counts(70_000) == (56_000, 7_000, 7_000)


def test_split_is_a_partition() -> None:
    payload = make_split_indices(70_000, seed=42)
    train, val, test = payload["train"], payload["val"], payload["test"]
    assert (len(train), len(val), len(test)) == (56_000, 7_000, 7_000)
    combined = train + val + test
    assert len(set(combined)) == 70_000
    assert set(combined) == set(range(70_000))
    assert payload["counts"] == {"train": 56_000, "val": 7_000, "test": 7_000}
    assert payload["seed"] == 42


def test_split_ignores_global_rng() -> None:
    torch.manual_seed(0)
    first = make_split_indices(1_000, seed=42)
    torch.manual_seed(7)
    _ = torch.rand(5)
    second = make_split_indices(1_000, seed=42)
    assert first == second
    assert canonical_digest(first) == canonical_digest(second)


def test_different_seed_changes_indices() -> None:
    first = make_split_indices(1_000, seed=42)
    second = make_split_indices(1_000, seed=43)
    assert first["train"] != second["train"]


def test_save_and_reload_are_identical(tmp_path) -> None:
    payload = make_split_indices(100, seed=42)
    path = tmp_path / "nested" / "split.json"
    save_split(payload, path)
    first = load_split(path)
    second = load_split(path)
    assert first == second == payload
    assert canonical_digest(first) == canonical_digest(payload)
