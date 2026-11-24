"""Train a decoder on fixed classifier features.

Each batch item is ``(z_MLP, image, label)``. Only the decoder is optimized.
Validation MSE selects the checkpoint. The test split is not read here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.classifier_config import ClassifierConfig
from models.classifier.frozen_decoder import FrozenDecoder


@dataclass
class DecoderTrainResult:
    best_epoch: int
    best_val_loss: float
    training_seconds: float
    history: list[dict]
    checkpoint_path: Path
    batch_size: int
    stopped_early: bool
    epochs_completed: int


def save_checkpoint(
    path: Path,
    model: FrozenDecoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    config: ClassifierConfig,
    history: list[dict],
    classifier_checkpoint: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_epoch": epoch,
        "best_validation_mse": val_loss,
        "config": config.to_dict(),
        "seed": config.seed,
        "history": history,
        "classifier_checkpoint": classifier_checkpoint,
        "classifier_frozen": True,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: FrozenDecoder,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def _run_epoch(
    model: FrozenDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    seen = 0
    gradient_mode = torch.enable_grad() if training else torch.no_grad()
    with gradient_mode:
        for latent, images, _labels in loader:
            latent = latent.to(device, non_blocking=device.type == "cuda")
            images = images.to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            reconstruction = model(latent)
            loss = F.mse_loss(reconstruction, images)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = int(images.shape[0])
            seen += batch_size
            total += float(loss.detach()) * batch_size
    if seen == 0:
        raise RuntimeError("the decoder loader produced zero images")
    return total / seen


def fit(
    model: FrozenDecoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ClassifierConfig,
    device: torch.device,
    checkpoint_path: Path,
    log_path: Path,
    classifier_checkpoint: str,
) -> DecoderTrainResult:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.decoder_learning_rate)
    history: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    waits = 0
    stopped_early = False
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0

    for epoch in range(1, config.decoder_max_epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = _run_epoch(model, train_loader, optimizer, device)
        val_loss = _run_epoch(model, val_loader, None, device)
        if not torch.isfinite(torch.tensor(train_loss)) or not torch.isfinite(torch.tensor(val_loss)):
            raise RuntimeError(f"non-finite decoder loss at epoch {epoch}")
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - epoch_started,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        completed = epoch
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            waits = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                best_val,
                config,
                history,
                classifier_checkpoint,
            )
        else:
            waits += 1
        log_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(
            f"frozen_decoder epoch {epoch:02d}/{config.decoder_max_epochs} "
            f"train_mse={train_loss:.6f} val_mse={val_loss:.6f} "
            f"seconds={record['seconds']:.2f} best_epoch={best_epoch}",
            flush=True,
        )
        if waits >= config.decoder_patience:
            stopped_early = True
            print(f"frozen_decoder early_stop at epoch {epoch}", flush=True)
            break

    if best_epoch == 0:
        raise RuntimeError("frozen decoder never stored a checkpoint")
    return DecoderTrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val,
        training_seconds=time.perf_counter() - started,
        history=history,
        checkpoint_path=checkpoint_path,
        batch_size=config.batch_size,
        stopped_early=stopped_early,
        epochs_completed=completed,
    )
