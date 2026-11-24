"""Smoke test for the Stage 1 data contract.

Run from the repository root:

    .venv/bin/python scripts/check_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import RandomSampler, SequentialSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import POOLED_SIZE, build_dataloaders, download_fashion_mnist
from data.splits import canonical_digest, load_split, make_split_indices, save_split

CONFIG_PATH = ROOT / "configs" / "data.yaml"
PREVIEW_PATH = ROOT / "outputs" / "figures" / "fashion_mnist_samples.png"
FASHION_CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = (
        "seed",
        "batch_size",
        "num_workers",
        "data_root",
        "split_path",
        "image_height",
        "image_width",
        "channels",
        "num_classes",
        "pixel_min",
        "pixel_max",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"config {CONFIG_PATH} is missing keys: {missing}")
    return config


def report_cuda() -> None:
    print(f"torch={torch.__version__}")
    print(f"cuda_compiled={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to this PyTorch build")
    device = torch.device("cuda")
    probe = torch.zeros(1, device=device)
    print(f"gpu_name={torch.cuda.get_device_name(0)}")
    print(f"gpu_memory_gb={torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f}")
    print(f"probe_device={probe.device}")


def ensure_split(path: Path, n: int, seed: int) -> dict:
    expected = make_split_indices(n, seed=seed)
    if path.exists():
        loaded = load_split(path)
        if canonical_digest(loaded) != canonical_digest(expected):
            raise SystemExit(
                f"existing split {path} does not match seed {seed}. "
                "Refusing to overwrite it."
            )
        print(f"split_file=reused path={path}")
    else:
        save_split(expected, path)
        loaded = load_split(path)
        print(f"split_file=created path={path}")
    if canonical_digest(loaded) != canonical_digest(expected):
        raise SystemExit("split file changed between save and reload")
    reloaded = load_split(path)
    if reloaded != loaded:
        raise SystemExit("two loads of the split file are not identical")
    print(f"split_digest={canonical_digest(loaded)}")
    print(
        "split_counts="
        f"train:{len(loaded['train'])} "
        f"val:{len(loaded['val'])} "
        f"test:{len(loaded['test'])}"
    )
    print(f"split_head_train={loaded['train'][:8]}")
    print(f"split_head_val={loaded['val'][:8]}")
    print(f"split_head_test={loaded['test'][:8]}")
    return loaded


def check_batch(name: str, batch: tuple[torch.Tensor, torch.Tensor], config: dict) -> None:
    images, labels = batch
    expected_shape = (
        images.shape[0],
        config["channels"],
        config["image_height"],
        config["image_width"],
    )
    if images.shape != expected_shape:
        raise SystemExit(f"{name} image shape {tuple(images.shape)} != {expected_shape}")
    if images.dtype != torch.float32:
        raise SystemExit(f"{name} image dtype is {images.dtype}, expected float32")
    if labels.dtype != torch.int64:
        raise SystemExit(f"{name} label dtype is {labels.dtype}, expected int64")
    if images.ndim != 4 or labels.shape != (images.shape[0],):
        raise SystemExit(f"{name} batch rank is wrong")
    minimum = float(images.min())
    maximum = float(images.max())
    if minimum < config["pixel_min"] or maximum > config["pixel_max"]:
        raise SystemExit(f"{name} pixels [{minimum}, {maximum}] leave [0, 1]")
    if int(labels.min()) < 0 or int(labels.max()) >= config["num_classes"]:
        raise SystemExit(f"{name} labels leave 0..{config['num_classes'] - 1}")
    print(
        f"batch_{name}=shape:{tuple(images.shape)} dtype:{images.dtype} "
        f"min:{minimum:.4f} max:{maximum:.4f} "
        f"labels:{tuple(int(v) for v in labels[:8])}"
    )


def save_sample_grid(images: torch.Tensor, labels: torch.Tensor, path: Path) -> None:
    """Save the first 16 test images as a 4x4 grid. Pixel values stay in [0, 1]."""
    if images.shape[0] < 16:
        raise SystemExit(f"need 16 images for the preview, got {images.shape[0]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, 4, figsize=(6, 6))
    for index, axis in enumerate(axes.ravel()):
        axis.imshow(images[index, 0].detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(FASHION_CLASSES[int(labels[index])], fontsize=8)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"preview was not written: {path}")
    print(f"preview={path} bytes={path.stat().st_size}")


def first_labels(loader) -> list[int]:
    _images, labels = next(iter(loader))
    return [int(value) for value in labels.tolist()]


def main() -> None:
    config = load_config()
    report_cuda()

    data_root = ROOT / config["data_root"]
    split_path = ROOT / config["split_path"]
    download_fashion_mnist(data_root)
    print(f"dataset_root={data_root}")
    print(f"pooled_size={POOLED_SIZE}")

    split = ensure_split(split_path, POOLED_SIZE, int(config["seed"]))
    covered = set(split["train"]) | set(split["val"]) | set(split["test"])
    if covered != set(range(POOLED_SIZE)):
        raise SystemExit("split is not a partition of 0..69999")
    if set(split["train"]) & set(split["val"]) or set(split["train"]) & set(split["test"]):
        raise SystemExit("split subsets overlap")
    if set(split["val"]) & set(split["test"]):
        raise SystemExit("validation and test overlap")

    loaders = build_dataloaders(
        data_root,
        split,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        seed=int(config["seed"]),
    )
    print(
        "loader_sizes="
        f"train:{len(loaders['train'].dataset)} "
        f"val:{len(loaders['val'].dataset)} "
        f"test:{len(loaders['test'].dataset)}"
    )
    if not isinstance(loaders["train"].sampler, RandomSampler):
        raise SystemExit("train loader is not shuffled")
    if not isinstance(loaders["val"].sampler, SequentialSampler):
        raise SystemExit("validation loader is shuffled")
    if not isinstance(loaders["test"].sampler, SequentialSampler):
        raise SystemExit("test loader is shuffled")
    print("shuffle=train:RandomSampler val:SequentialSampler test:SequentialSampler")
    train_first_labels: list[int] | None = None
    for name in ("train", "val", "test"):
        images, labels = next(iter(loaders[name]))
        check_batch(name, (images, labels), config)
        if name == "train":
            train_first_labels = [int(value) for value in labels.tolist()]
        if name == "test":
            save_sample_grid(images, labels, PREVIEW_PATH)

    val_a = first_labels(loaders["val"])
    val_b = first_labels(loaders["val"])
    test_a = first_labels(loaders["test"])
    test_b = first_labels(loaders["test"])
    if val_a != val_b or test_a != test_b:
        raise SystemExit("validation or test order changed across iterations")
    print("val_test_order=stable")

    rebuilt = build_dataloaders(
        data_root,
        split,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        seed=int(config["seed"]),
    )
    rebuilt_first = first_labels(rebuilt["train"])
    if train_first_labels != rebuilt_first:
        raise SystemExit("seeded train shuffle was not reproduced")
    print("train_shuffle=reproducible_with_seed_42")
    print("smoke_test=ok")


if __name__ == "__main__":
    main()
