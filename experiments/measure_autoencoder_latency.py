"""Batch-64 inference latency for the three trained autoencoders.

    .venv/bin/python experiments/measure_autoencoder_latency.py

The timed region is the model forward only. Data loading and checkpoint
loading happen before the clock starts. VAE timing uses the posterior mean,
then the decoder, so the measurement does not include a random sample.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path

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
from training.autoencoder_trainer import load_checkpoint

CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "autoencoders"
LATENCY_PATH = ROOT / "outputs" / "metrics" / "autoencoder_latency.json"
COMPARISON_PATH = ROOT / "outputs" / "metrics" / "model_comparison.csv"
WARMUP = 10
REPEATS = 100
BATCH_SIZE = 64
NAMES = {
    "Vanilla AE": ("vanilla_ae_best.pt", VanillaAE),
    "Sparse AE": ("sparse_ae_best.pt", SparseAE),
    "VAE": ("vae_best.pt", VariationalAE),
}


def reconstruct(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    if isinstance(model, VariationalAE):
        mu, _logvar = model.encode(images)
        return model.decode(mu)
    return model(images)[0]


def time_model(model: torch.nn.Module, images: torch.Tensor, device: torch.device) -> dict:
    model.eval()
    with torch.inference_mode():
        for _ in range(WARMUP):
            reconstruct(model, images)
            if device.type == "cuda":
                torch.cuda.synchronize()
        samples = []
        for _ in range(REPEATS):
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            reconstruct(model, images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - started)
    mean_seconds = statistics.fmean(samples)
    return {
        "mean_seconds": mean_seconds,
        "median_seconds": statistics.median(samples),
        "std_seconds": statistics.stdev(samples),
        "mean_milliseconds": mean_seconds * 1000.0,
        "median_milliseconds": statistics.median(samples) * 1000.0,
        "std_milliseconds": statistics.stdev(samples) * 1000.0,
        "milliseconds_per_image": mean_seconds * 1000.0 / BATCH_SIZE,
    }


def update_comparison(latency_ms: dict[str, float]) -> None:
    with COMPARISON_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        if row["Model"] in latency_ms:
            row["Latency ms Batch 64"] = f"{latency_ms[row['Model']]:.4f}"
    with COMPARISON_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with (ROOT / "configs" / "data.yaml").open(encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle)
    split = load_split(ROOT / data_config["split_path"])
    loader = build_dataloaders(
        ROOT / data_config["data_root"],
        split,
        batch_size=BATCH_SIZE,
        num_workers=0,
        seed=int(data_config["seed"]),
    )["test"]
    images, _labels = next(iter(loader))
    images = images[:BATCH_SIZE].to(device)
    if images.shape[0] != BATCH_SIZE:
        raise RuntimeError(f"expected {BATCH_SIZE} test images, got {images.shape[0]}")

    report = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": BATCH_SIZE,
        "warmup": WARMUP,
        "iterations": REPEATS,
        "std_estimator": "sample standard deviation of the iteration times",
        "timed_region": "model forward only; data loading and checkpoint loading are excluded",
        "vae_method": "encode to posterior mean mu, then decode. Reparameterization sampling is not included.",
        "ae_method": "encoder plus decoder through the model forward",
        "models": {},
    }
    comparison_ms = {}
    for label, (filename, factory) in NAMES.items():
        model = factory(64)
        load_checkpoint(CHECKPOINT_DIR / filename, model, map_location=device)
        model.to(device)
        measured = time_model(model, images, device)
        report["models"][label] = measured
        comparison_ms[label] = measured["mean_milliseconds"]
        print(
            f"{label} mean_ms={measured['mean_milliseconds']:.4f} "
            f"median_ms={measured['median_milliseconds']:.4f} "
            f"std_ms={measured['std_milliseconds']:.4f} "
            f"per_image_ms={measured['milliseconds_per_image']:.6f}",
            flush=True,
        )
    LATENCY_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    update_comparison(comparison_ms)
    print(f"latency={LATENCY_PATH}", flush=True)


if __name__ == "__main__":
    main()
