"""Train Vanilla GAN, then DCGAN, then write the stage-3 comparison files.

    .venv/bin/python experiments/run_gans.py

The test split is not loaded. Validation is used only to measure discriminator
accuracy after each epoch.
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

from configs.gan_config import GanConfig, load_gan_config
from data.dataset import build_dataloaders
from data.splits import canonical_digest, load_split
from evaluation.fid import pretrained_inception_status
from evaluation.inference_latency import measure_generator_latency
from models.gans.dcgan import DCGANDiscriminator, DCGANGenerator
from models.gans.vanilla_gan import VanillaDiscriminator, VanillaGenerator
from training.gan_trainer import (
    GanTrainResult,
    fit_gan,
    generate_images,
    load_checkpoint,
    make_fixed_noise,
)
from visualization.figure_02_generation import save_figure_02
from visualization.gan_training_curves import save_gan_curves, save_progress_figure, snapshots_from_checkpoints

CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "gans"
LOG_DIR = ROOT / "outputs" / "logs"
SAMPLE_DIR = ROOT / "outputs" / "figures" / "gan_samples"
METRICS_DIR = ROOT / "outputs" / "metrics"
MODEL_ORDER = ("vanilla_gan", "dcgan")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_pair(name: str, latent_dim: int) -> tuple[torch.nn.Module, torch.nn.Module]:
    if name == "vanilla_gan":
        return VanillaGenerator(latent_dim), VanillaDiscriminator()
    if name == "dcgan":
        return DCGANGenerator(latent_dim), DCGANDiscriminator()
    raise ValueError(name)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def make_loaders(config: GanConfig, split: dict) -> dict:
    return build_dataloaders(
        ROOT / config.data_root,
        split,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )


def train_model(name: str, config: GanConfig, split: dict, device: torch.device) -> tuple[GanTrainResult, int, int]:
    active = config
    for attempt in (1, 2):
        set_seed(active.seed)
        generator, discriminator = build_pair(name, active.latent_dim)
        generator_params = count_parameters(generator)
        discriminator_params = count_parameters(discriminator)
        fixed_noise = make_fixed_noise(active.fixed_noise_count, active.latent_dim, active.seed)
        loaders = make_loaders(active, split)
        try:
            result = fit_gan(
                generator,
                discriminator,
                loaders["train"],
                loaders["val"],
                active,
                device,
                CHECKPOINT_DIR,
                LOG_DIR / f"{name}_history.json",
                SAMPLE_DIR,
                name,
                fixed_noise,
            )
            return result, generator_params, discriminator_params
        except torch.cuda.OutOfMemoryError:
            if attempt == 2 or active.batch_size <= 64:
                raise
            torch.cuda.empty_cache()
            active = replace(active, batch_size=64)
            print(f"{name} CUDA out of memory; retrying with batch_size=64", flush=True)
    raise RuntimeError(f"{name} did not train")


def main() -> None:
    config = load_gan_config(ROOT)
    if config.seed != 42 or config.latent_dim != 100 or config.batch_size != 128:
        raise SystemExit(f"locked GAN config changed: {config}")
    split = load_split(ROOT / config.split_path)
    digest = canonical_digest(split)
    if digest != "963816f417b1122c50c162765938984622f50669a4401651192ff30320d55b12":
        raise SystemExit("refusing to train because the split digest changed")
    if (len(split["train"]), len(split["val"]), len(split["test"])) != (56000, 7000, 7000):
        raise SystemExit("refusing to train on a split other than 56000/7000/7000")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    trained: dict[str, GanTrainResult] = {}
    parameter_counts: dict[str, dict[str, int]] = {}
    for name in MODEL_ORDER:
        result, generator_params, discriminator_params = train_model(name, config, split, device)
        trained[name] = result
        parameter_counts[name] = {
            "generator": generator_params,
            "discriminator": discriminator_params,
        }

    fid_status = pretrained_inception_status()
    rows = []
    detailed = {}
    histories = {}
    generated = {}
    for name in MODEL_ORDER:
        result = trained[name]
        generator, _discriminator = build_pair(name, config.latent_dim)
        payload = load_checkpoint(result.final_checkpoint, generator, map_location=device)
        generator.to(device)
        generator.eval()
        latency = measure_generator_latency(generator, device, batch_size=64, warmup=10, repeats=30)
        images = generate_images(generator, payload["fixed_noise"][:16], device)
        generated[name] = images.cpu()
        final = result.history[-1]
        row = {
            "Model": name,
            "Latent Dim": config.latent_dim,
            "Epochs": result.epochs_completed,
            "Training Seconds": f"{result.training_seconds:.2f}",
            "Final Generator Loss": f"{final['generator_loss']:.6f}",
            "Final Discriminator Loss": f"{final['discriminator_loss']:.6f}",
            "Final Real Accuracy": f"{final['discriminator_real_accuracy']:.6f}",
            "Final Fake Accuracy": f"{final['discriminator_fake_accuracy']:.6f}",
            "Latency ms": f"{latency['mean_milliseconds']:.4f}",
            "Generator Params": parameter_counts[name]["generator"],
            "Discriminator Params": parameter_counts[name]["discriminator"],
            "FID": "unavailable" if not fid_status["available"] else "not_run",
        }
        rows.append(row)
        histories[name] = result.history
        detailed[name] = {
            "epochs_completed": result.epochs_completed,
            "training_seconds": result.training_seconds,
            "batch_size": result.batch_size,
            "checkpoint": str(result.final_checkpoint.relative_to(ROOT)),
            "parameters": parameter_counts[name],
            "final_epoch": final,
            "latency": latency,
            "history": result.history,
        }
        print(
            f"{name} epochs={result.epochs_completed} seconds={result.training_seconds:.2f} "
            f"G={final['generator_loss']:.4f} D={final['discriminator_loss']:.4f} "
            f"latency_ms={latency['mean_milliseconds']:.4f}",
            flush=True,
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with (METRICS_DIR / "gan_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "config": config.to_dict(),
        "split_digest": digest,
        "split_counts": {"train": 56000, "val": 7000, "test": 7000},
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "loss_definition": (
            "BCEWithLogitsLoss reduction=mean. Discriminator loss is the sum of the "
            "real-term mean and the fake-term mean. Generator loss is the mean BCE "
            "of fake logits against target 1."
        ),
        "pixel_transform": "dataset [0, 1] mapped once with x*2-1; figures use (x+1)/2",
        "fid": fid_status,
        "models": detailed,
    }
    (METRICS_DIR / "gan_comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    save_figure_02(generated["vanilla_gan"], generated["dcgan"])
    save_gan_curves(histories)
    save_progress_figure(snapshots_from_checkpoints(device))
    print(f"csv={METRICS_DIR / 'gan_comparison.csv'}", flush=True)
    print(f"fid_available={fid_status['available']}", flush=True)


if __name__ == "__main__":
    main()
