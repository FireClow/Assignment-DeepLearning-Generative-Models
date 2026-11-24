"""Train Vanilla AE, Sparse AE, and VAE, then score the held-out test split.

    .venv/bin/python experiments/run_autoencoders.py

The test loader is opened only after every checkpoint has been chosen from
validation loss. A CUDA out-of-memory error retries that model at batch size 64.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.autoencoder_config import AutoencoderConfig, load_autoencoder_config
from data.dataset import build_dataloaders
from data.splits import canonical_digest, load_split
from evaluation.reconstruction_metrics import evaluate_reconstruction
from models.autoencoders.sparse_ae import SparseAE
from models.autoencoders.vae import VariationalAE
from models.autoencoders.vanilla_ae import VanillaAE
from training.autoencoder_trainer import TrainResult, fit, load_checkpoint, resolve_device
from visualization.figure_01_reconstruction import first_test_images, save_reconstruction_grid
from visualization.plot_utils import save_training_curves

CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "autoencoders"
LOG_DIR = ROOT / "outputs" / "logs"
METRICS_DIR = ROOT / "outputs" / "metrics"
CSV_PATH = METRICS_DIR / "autoencoder_comparison.csv"
JSON_PATH = METRICS_DIR / "autoencoder_comparison.json"
CURVE_PATH = ROOT / "outputs" / "figures" / "autoencoder_training_curves.png"
MODEL_ORDER = ("vanilla_ae", "sparse_ae", "vae")
CHECKPOINT_NAMES = {
    "vanilla_ae": "vanilla_ae_best.pt",
    "sparse_ae": "sparse_ae_best.pt",
    "vae": "vae_best.pt",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str, latent_dim: int) -> torch.nn.Module:
    if name == "vanilla_ae":
        return VanillaAE(latent_dim)
    if name == "sparse_ae":
        return SparseAE(latent_dim)
    if name == "vae":
        return VariationalAE(latent_dim)
    raise ValueError(f"unknown model {name}")


def make_loaders(config: AutoencoderConfig, split: dict) -> dict:
    return build_dataloaders(
        ROOT / config.data_root,
        split,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )


def train_model(name: str, config: AutoencoderConfig, split: dict, device: torch.device) -> TrainResult:
    active = config
    for attempt in (1, 2):
        set_seed(active.seed)
        model = build_model(name, active.latent_dim)
        loaders = make_loaders(active, split)
        try:
            return fit(
                model,
                loaders["train"],
                loaders["val"],
                active,
                device,
                CHECKPOINT_DIR / CHECKPOINT_NAMES[name],
                LOG_DIR / f"{name}_history.json",
                name,
            )
        except torch.cuda.OutOfMemoryError:
            if attempt == 2 or active.batch_size <= 64:
                raise
            torch.cuda.empty_cache()
            active = replace(active, batch_size=64)
            print(f"{name} CUDA out of memory; retrying with batch_size=64", flush=True)
    raise RuntimeError(f"{name} did not train")


def write_reports(config: AutoencoderConfig, split: dict, rows: list[dict], results: dict) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Model",
        "Latent Dim",
        "Test MSE",
        "Test PSNR",
        "Test SSIM",
        "Best Epoch",
        "Training Time",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "config": config.to_dict(),
        "split_digest": canonical_digest(split),
        "split_counts": {
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models": results,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    config = load_autoencoder_config(ROOT)
    if config.seed != 42 or config.latent_dim != 64 or config.batch_size != 128:
        raise SystemExit(f"locked config changed: {config}")
    split = load_split(ROOT / config.split_path)
    if (len(split["train"]), len(split["val"]), len(split["test"])) != (56000, 7000, 7000):
        raise SystemExit("refusing to train on a split other than 56000/7000/7000")
    device = resolve_device()
    print(f"device={device}", flush=True)

    trained: dict[str, TrainResult] = {}
    for name in MODEL_ORDER:
        trained[name] = train_model(name, config, split, device)

    set_seed(config.seed)
    test_loader = make_loaders(config, split)["test"]
    rows = []
    detailed = {}
    histories = {}
    for name in MODEL_ORDER:
        result = trained[name]
        set_seed(config.seed)
        model = build_model(name, config.latent_dim)
        load_checkpoint(result.checkpoint_path, model, map_location=device)
        model.to(device)
        model.eval()
        metrics = evaluate_reconstruction(model, test_loader, device)
        row = {
            "Model": name,
            "Latent Dim": config.latent_dim,
            "Test MSE": f"{metrics['mse']:.8f}",
            "Test PSNR": f"{metrics['psnr']:.8f}",
            "Test SSIM": f"{metrics['ssim']:.8f}",
            "Best Epoch": result.best_epoch,
            "Training Time": f"{result.training_seconds:.2f}",
        }
        rows.append(row)
        histories[name] = result.history
        detailed[name] = {
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "training_seconds": result.training_seconds,
            "batch_size": result.batch_size,
            "checkpoint": str(result.checkpoint_path.relative_to(ROOT)),
            "test_metrics": metrics,
            "history": result.history,
        }
        print(
            f"{name} test_mse={metrics['mse']:.8f} "
            f"test_psnr={metrics['psnr']:.8f} test_ssim={metrics['ssim']:.8f} "
            f"images={int(metrics['num_images'])}",
            flush=True,
        )

    write_reports(config, split, rows, detailed)
    save_training_curves(histories, CURVE_PATH)
    images, labels = first_test_images()
    images = images.to(device)
    reconstructions = {}
    with torch.no_grad():
        for name in MODEL_ORDER:
            model = build_model(name, config.latent_dim)
            load_checkpoint(trained[name].checkpoint_path, model, map_location=device)
            model.to(device)
            model.eval()
            reconstructions[name] = model(images)[0].clamp(0.0, 1.0)
    save_reconstruction_grid(images, labels, reconstructions)
    print(f"csv={CSV_PATH}", flush=True)
    print(f"json={JSON_PATH}", flush=True)
    print(f"curves={CURVE_PATH}", flush=True)


if __name__ == "__main__":
    main()
