"""Train the DDPM, then score noise prediction on the held-out test split.

    .venv/bin/python experiments/run_ddpm.py

The test loader is read only after the checkpoint has been chosen from
validation noise-prediction MSE. Sampling for the figures uses that checkpoint
and 1,000 reverse steps.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.ddpm_config import DDPMConfig, load_ddpm_config
from data.dataset import build_dataloaders
from data.splits import canonical_digest, load_split
from evaluation.ddpm_metrics import fixed_seed_noise_mse
from evaluation.fid import pretrained_inception_status
from evaluation.inference_latency import measure_ddpm_latency
from models.autoencoders.vae import VariationalAE
from models.diffusion.ddpm import TRAJECTORY_LEVELS, diffusion_to_unit
from training.ddpm_trainer import DDPMTrainResult, build_ddpm, fit, load_checkpoint
from training.gan_trainer import generate_images
from visualization.ddpm_trajectory import save_trajectory
from visualization.figure_02_generation import (
    CHECKPOINT_DIR as GAN_CHECKPOINT_DIR,
    load_generator,
    save_figure_02_comparison,
)
from visualization.figure_05_training_curves import save_figure_05
from visualization.plot_utils import save_gray_grid

SPLIT_DIGEST = "963816f417b1122c50c162765938984622f50669a4401651192ff30320d55b12"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "diffusion"
LOG_DIR = ROOT / "outputs" / "logs"
METRICS_DIR = ROOT / "outputs" / "metrics"
FIGURE_DIR = ROOT / "outputs" / "figures"
CSV_PATH = METRICS_DIR / "ddpm_metrics.csv"
JSON_PATH = METRICS_DIR / "ddpm_metrics.json"
FIGURE_02 = FIGURE_DIR / "figure_02_generation.png"
FIGURE_02_BACKUP = FIGURE_DIR / "figure_02_generation_gans.png"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def make_loaders(config: DDPMConfig, split: dict) -> dict:
    return build_dataloaders(
        ROOT / config.data_root,
        split,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )


def train_model(config: DDPMConfig, split: dict, device: torch.device) -> tuple[DDPMTrainResult, int]:
    active = config
    for attempt in (1, 2):
        set_seed(active.seed)
        model = build_ddpm(active)
        parameters = count_parameters(model.unet)
        loaders = make_loaders(active, split)
        try:
            result = fit(
                model,
                loaders["train"],
                loaders["val"],
                active,
                device,
                CHECKPOINT_DIR,
                LOG_DIR / "ddpm_history.json",
            )
            return result, parameters
        except torch.cuda.OutOfMemoryError:
            if attempt == 2 or active.batch_size <= 64:
                raise
            torch.cuda.empty_cache()
            active = replace(active, batch_size=64)
            print("ddpm CUDA out of memory; retrying with batch_size=64", flush=True)
    raise RuntimeError("ddpm did not train")


def vae_prior_samples(device: torch.device) -> torch.Tensor:
    payload = torch.load(
        ROOT / "outputs" / "checkpoints" / "autoencoders" / "vae_best.pt",
        map_location=device,
        weights_only=False,
    )
    latent_dim = int(payload["config"]["latent_dim"])
    model = VariationalAE(latent_dim)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    generator = torch.Generator()
    generator.manual_seed(42)
    latent = torch.randn(16, latent_dim, generator=generator).to(device)
    with torch.no_grad():
        images = model.decode(latent)
    return images.detach().cpu()


def gan_samples(name: str, device: torch.device) -> torch.Tensor:
    generator, noise = load_generator(name, device)
    return generate_images(generator, noise[:16].to(device), device).detach().cpu()


def main() -> None:
    config = load_ddpm_config(ROOT)
    if (
        config.seed != 42
        or config.batch_size != 128
        or config.timesteps != 1000
        or config.beta_start != 1e-4
        or config.beta_end != 0.02
        or config.learning_rate != 2e-4
        or config.max_epochs != 30
        or config.patience != 5
    ):
        raise SystemExit(f"locked DDPM config changed: {config}")
    split = load_split(ROOT / config.split_path)
    digest = canonical_digest(split)
    if digest != SPLIT_DIGEST:
        raise SystemExit("refusing to train because the split digest changed")
    if (len(split["train"]), len(split["val"]), len(split["test"])) != (56000, 7000, 7000):
        raise SystemExit("refusing to train on a split other than 56000/7000/7000")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    result, parameters = train_model(config, split, device)
    loaders = make_loaders(
        replace(config, batch_size=result.batch_size),
        split,
    )
    model = build_ddpm(replace(config, batch_size=result.batch_size))
    load_checkpoint(result.best_checkpoint, model, map_location=device)
    model.to(device)
    test_metrics = fixed_seed_noise_mse(
        model,
        loaders["test"],
        device,
        config.validation_seed,
        "test",
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    latency = measure_ddpm_latency(model, device, batch_size=64, warmup=1, repeats=3)
    if latency["reverse_steps"] != 1000:
        raise RuntimeError(f"latency used {latency['reverse_steps']} steps, expected 1000")

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    frames: dict[int, torch.Tensor] = {}
    samples = model.sample(16, device, generator=generator, frames=frames)
    ddpm_images = diffusion_to_unit(samples).detach().cpu()
    missing = [level for level in TRAJECTORY_LEVELS if level not in frames]
    if missing:
        raise RuntimeError(f"trajectory missing levels {missing}")
    trajectory = {level: diffusion_to_unit(frames[level][:1]) for level in TRAJECTORY_LEVELS}

    if FIGURE_02.is_file() and not FIGURE_02_BACKUP.is_file():
        shutil.copy2(FIGURE_02, FIGURE_02_BACKUP)
    save_figure_02_comparison(
        vae_prior_samples(device),
        gan_samples("vanilla_gan", device),
        gan_samples("dcgan", device),
        ddpm_images,
        FIGURE_02,
    )
    save_trajectory(trajectory)
    save_figure_05(trajectory)
    save_gray_grid(ddpm_images, FIGURE_DIR / "ddpm_samples.png", "DDPM samples, 1000 reverse steps")

    fid_status = pretrained_inception_status()
    fid_value = "N/A"
    still_best_at_end = result.best_epoch == result.epochs_completed
    record = {
        "model": "ddpm",
        "epochs_completed": result.epochs_completed,
        "best_epoch": result.best_epoch,
        "stopped_early": result.stopped_early,
        "validation_still_best_on_last_epoch": still_best_at_end,
        "best_val_noise_prediction_mse": result.best_val_loss,
        "test_noise_prediction_mse": test_metrics["test_noise_prediction_mse"],
        "test_images": test_metrics["images"],
        "noise_metric": "noise_prediction_mse",
        "training_seconds": result.training_seconds,
        "batch_size": result.batch_size,
        "learning_rate": config.learning_rate,
        "grad_clip": config.grad_clip,
        "timesteps": config.timesteps,
        "beta_schedule": config.beta_schedule,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "parameters": parameters,
        "latency": latency,
        "fid": fid_value,
        "fid_status": fid_status,
        "split_digest": digest,
        "checkpoints": {
            "best": str(result.best_checkpoint),
            "final": str(result.final_checkpoint),
        },
        "history": result.history,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Model",
                "Epochs",
                "Best Epoch",
                "Best Val Noise MSE",
                "Test Noise MSE",
                "Training Seconds",
                "Parameters",
                "Latency ms",
                "Reverse Steps",
                "FID",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Model": "ddpm",
                "Epochs": result.epochs_completed,
                "Best Epoch": result.best_epoch,
                "Best Val Noise MSE": f"{result.best_val_loss:.6f}",
                "Test Noise MSE": f"{test_metrics['test_noise_prediction_mse']:.6f}",
                "Training Seconds": f"{result.training_seconds:.2f}",
                "Parameters": parameters,
                "Latency ms": f"{latency['mean_milliseconds']:.4f}",
                "Reverse Steps": latency["reverse_steps"],
                "FID": fid_value,
            }
        )
    print(
        f"ddpm epochs={result.epochs_completed} best_epoch={result.best_epoch} "
        f"best_val_noise_mse={result.best_val_loss:.6f} "
        f"test_noise_mse={test_metrics['test_noise_prediction_mse']:.6f} "
        f"seconds={result.training_seconds:.2f} params={parameters} "
        f"latency_ms={latency['mean_milliseconds']:.4f} steps={latency['reverse_steps']} "
        f"fid={fid_value} early_stop={result.stopped_early} "
        f"last_epoch_is_best={still_best_at_end}",
        flush=True,
    )
    print(f"csv={CSV_PATH}", flush=True)
    print(f"gan_checkpoint_dir={GAN_CHECKPOINT_DIR}", flush=True)


if __name__ == "__main__":
    main()
