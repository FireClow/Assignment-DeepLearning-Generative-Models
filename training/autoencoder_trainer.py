"""Adam training for the three autoencoders.

The checkpoint is chosen by validation total loss. The test loader is not
accepted by fit(), so it cannot influence early stopping.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from configs.autoencoder_config import AutoencoderConfig
from models.autoencoders.sparse_ae import SparseAE, sparse_ae_loss
from models.autoencoders.vae import VariationalAE, vae_loss
from models.autoencoders.vanilla_ae import VanillaAE, mse_reconstruction


@dataclass
class TrainResult:
    model_name: str
    best_epoch: int
    best_val_loss: float
    training_seconds: float
    history: list[dict]
    checkpoint_path: Path
    batch_size: int


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_losses(
    model: nn.Module,
    images: torch.Tensor,
    config: AutoencoderConfig,
) -> dict[str, torch.Tensor]:
    """Return batch-mean total, reconstruction, and penalty scalars.

    The penalty is 0 for Vanilla AE, mean(|z|) for Sparse AE, and KL for VAE.
    It is not multiplied by lambda or beta; that weight is inside the total.
    """
    if isinstance(model, VariationalAE):
        reconstruction, mu, logvar, _latent = model(images)
        total, reconstruction_loss, kl_loss = vae_loss(
            reconstruction, images, mu, logvar, beta=config.vae_beta
        )
        return {"total": total, "reconstruction": reconstruction_loss, "penalty": kl_loss}
    if isinstance(model, SparseAE):
        reconstruction, latent = model(images)
        total, reconstruction_loss, sparsity_loss = sparse_ae_loss(
            reconstruction, images, latent, lambda_sparse=config.sparse_lambda
        )
        return {
            "total": total,
            "reconstruction": reconstruction_loss,
            "penalty": sparsity_loss,
        }
    if isinstance(model, VanillaAE):
        reconstruction, _latent = model(images)
        reconstruction_loss = mse_reconstruction(reconstruction, images)
        zero = torch.zeros((), device=images.device)
        return {"total": reconstruction_loss, "reconstruction": reconstruction_loss, "penalty": zero}
    raise TypeError(f"unsupported autoencoder type: {type(model).__name__}")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    config: AutoencoderConfig,
    model_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "config": config.to_dict(),
        "seed": config.seed,
        "model_name": model_name,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load weights saved by save_checkpoint. Optimizer state is optional."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    config: AutoencoderConfig,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"reconstruction": 0.0, "penalty": 0.0, "total": 0.0}
    seen = 0
    gradient_mode = torch.enable_grad() if training else torch.no_grad()
    with gradient_mode:
        for images, _labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            losses = compute_losses(model, images, config)
            if training:
                losses["total"].backward()
                optimizer.step()
            batch_size = int(images.shape[0])
            seen += batch_size
            for key in totals:
                totals[key] += float(losses[key].detach()) * batch_size
    if seen == 0:
        raise RuntimeError("the loader produced zero images")
    return {key: value / seen for key, value in totals.items()}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: AutoencoderConfig,
    device: torch.device,
    checkpoint_path: Path,
    log_path: Path,
    model_name: str,
) -> TrainResult:
    """Train until max_epochs or until validation loss stalls for `patience` epochs."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    waits = 0
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = _run_epoch(model, train_loader, optimizer, device, config)
        val_metrics = _run_epoch(model, val_loader, None, device, config)
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - epoch_started,
            "train_reconstruction": train_metrics["reconstruction"],
            "train_penalty": train_metrics["penalty"],
            "train_total": train_metrics["total"],
            "val_reconstruction": val_metrics["reconstruction"],
            "val_penalty": val_metrics["penalty"],
            "val_total": val_metrics["total"],
        }
        history.append(record)
        improved = val_metrics["total"] < best_val
        if improved:
            best_val = val_metrics["total"]
            best_epoch = epoch
            waits = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                best_val,
                config,
                model_name,
            )
        else:
            waits += 1
        log_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(
            f"{model_name} epoch {epoch:02d}/{config.max_epochs} "
            f"train_total={record['train_total']:.6f} "
            f"train_recon={record['train_reconstruction']:.6f} "
            f"train_penalty={record['train_penalty']:.6f} "
            f"val_total={record['val_total']:.6f} "
            f"val_recon={record['val_reconstruction']:.6f} "
            f"val_penalty={record['val_penalty']:.6f} "
            f"seconds={record['seconds']:.2f} best_epoch={best_epoch}",
            flush=True,
        )
        if waits >= config.patience:
            print(f"{model_name} early_stop at epoch {epoch}", flush=True)
            break

    if best_epoch == 0:
        raise RuntimeError(f"{model_name} never stored a checkpoint")
    return TrainResult(
        model_name=model_name,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        training_seconds=time.perf_counter() - started,
        history=history,
        checkpoint_path=checkpoint_path,
        batch_size=config.batch_size,
    )
