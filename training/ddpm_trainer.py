"""Adam training for DDPM noise prediction.

The checkpoint is chosen by validation noise-prediction MSE. ``fit`` does not
accept a test loader. Validation redraws timesteps from a private generator
reseeded with ``config.validation_seed``, so the score can be compared across
epochs without consuming the training RNG.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.ddpm_config import DDPMConfig
from evaluation.ddpm_metrics import fixed_seed_noise_mse
from models.diffusion.ddpm import DDPM, unit_to_diffusion
from models.diffusion.noise_scheduler import NoiseScheduler
from models.diffusion.unet import UNet


@dataclass
class DDPMTrainResult:
    epochs_completed: int
    best_epoch: int
    best_val_loss: float
    training_seconds: float
    history: list[dict]
    best_checkpoint: Path
    final_checkpoint: Path
    batch_size: int
    stopped_early: bool


def build_ddpm(config: DDPMConfig) -> DDPM:
    unet = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
        time_dim=config.time_dim,
    )
    scheduler = NoiseScheduler(
        timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        schedule=config.beta_schedule,
    )
    return DDPM(unet, scheduler)


def save_checkpoint(
    path: Path,
    model: DDPM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int,
    best_val_loss: float,
    waits: int,
    history: list[dict],
    config: DDPMConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "unet_state_dict": model.unet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "epochs_without_improvement": waits,
        "history": history,
        "scheduler_config": {
            "timesteps": config.timesteps,
            "beta_start": config.beta_start,
            "beta_end": config.beta_end,
            "schedule": config.beta_schedule,
        },
        "model_config": {
            "in_channels": config.in_channels,
            "out_channels": config.out_channels,
            "base_channels": config.base_channels,
            "time_dim": config.time_dim,
        },
        "config": config.to_dict(),
        "seed": config.seed,
        "diffusion_timesteps": config.timesteps,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: DDPM,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a U-Net trained by ``save_checkpoint``. The scheduler is rebuilt from config."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.unet.load_state_dict(payload["unet_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def _train_epoch(
    model: DDPM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None,
) -> float:
    model.unet.train()
    total = 0.0
    seen = 0
    for images, _labels in loader:
        images = unit_to_diffusion(images.to(device, non_blocking=device.type == "cuda"))
        optimizer.zero_grad(set_to_none=True)
        loss = model.noise_prediction_loss(images)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), grad_clip)
        optimizer.step()
        batch_size = int(images.shape[0])
        total += float(loss.detach()) * batch_size
        seen += batch_size
    if seen == 0:
        raise RuntimeError("the training loader produced zero images")
    return total / seen


def fit(
    model: DDPM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: DDPMConfig,
    device: torch.device,
    checkpoint_dir: Path,
    log_path: Path,
    resume_path: Path | None = None,
) -> DDPMTrainResult:
    """Train until ``max_epochs`` or until validation loss stalls for ``patience`` epochs."""
    model.to(device)
    optimizer = torch.optim.Adam(model.unet.parameters(), lr=config.learning_rate)
    history: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    waits = 0
    start_epoch = 1
    if resume_path is not None:
        payload = load_checkpoint(resume_path, model, optimizer, map_location=device)
        history = list(payload["history"])
        best_val = float(payload["best_validation_loss"])
        best_epoch = int(payload["best_epoch"])
        waits = int(payload["epochs_without_improvement"])
        start_epoch = int(payload["epoch"]) + 1
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "ddpm_best.pt"
    final_path = checkpoint_dir / "ddpm_final.pt"
    started = time.perf_counter()
    stopped_early = False
    completed = start_epoch - 1

    for epoch in range(start_epoch, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = _train_epoch(model, train_loader, optimizer, device, config.grad_clip)
        val_metrics = fixed_seed_noise_mse(
            model,
            val_loader,
            device,
            config.validation_seed,
            "val",
        )
        val_loss = float(val_metrics["val_noise_prediction_mse"])
        if not torch.isfinite(torch.tensor(train_loss)) or not torch.isfinite(torch.tensor(val_loss)):
            raise RuntimeError(f"non-finite noise loss at epoch {epoch}")
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - epoch_started,
            "train_noise_loss": train_loss,
            "val_noise_loss": val_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        completed = epoch
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            waits = 0
            save_checkpoint(
                best_path, model, optimizer, epoch, best_epoch, best_val, waits, history, config
            )
        else:
            waits += 1
        log_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(
            f"ddpm epoch {epoch:02d}/{config.max_epochs} "
            f"train_noise_mse={train_loss:.6f} val_noise_mse={val_loss:.6f} "
            f"lr={record['learning_rate']:.6g} seconds={record['seconds']:.2f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
        if waits >= config.patience:
            stopped_early = True
            print(f"ddpm early_stop at epoch {epoch}", flush=True)
            break

    if best_epoch == 0:
        raise RuntimeError("ddpm never stored a checkpoint")
    save_checkpoint(
        final_path, model, optimizer, completed, best_epoch, best_val, waits, history, config
    )
    return DDPMTrainResult(
        epochs_completed=completed,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        training_seconds=time.perf_counter() - started,
        history=history,
        best_checkpoint=best_path,
        final_checkpoint=final_path,
        batch_size=config.batch_size,
        stopped_early=stopped_early,
    )
