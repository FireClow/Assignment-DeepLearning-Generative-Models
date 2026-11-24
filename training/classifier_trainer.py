"""Adam training for the MLP classifier.

The checkpoint is chosen by validation cross-entropy. ``fit`` does not accept
a test loader. Accuracy is the fraction of argmax logits that match the label.
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
from models.classifier.mlp_classifier import MLPClassifier


@dataclass
class ClassifierTrainResult:
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
    model: MLPClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    config: ClassifierConfig,
    history: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_epoch": epoch,
        "best_validation_loss": val_loss,
        "config": config.to_dict(),
        "seed": config.seed,
        "history": history,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: MLPClassifier,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def _run_epoch(
    model: MLPClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    correct = 0.0
    seen = 0
    gradient_mode = torch.enable_grad() if training else torch.no_grad()
    with gradient_mode:
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = int(images.shape[0])
            seen += batch_size
            loss_total += float(loss.detach()) * batch_size
            correct += float((logits.argmax(dim=1) == labels).sum().item())
    if seen == 0:
        raise RuntimeError("the loader produced zero images")
    return {"loss": loss_total / seen, "accuracy": correct / seen}


def fit(
    model: MLPClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ClassifierConfig,
    device: torch.device,
    checkpoint_path: Path,
    log_path: Path,
) -> ClassifierTrainResult:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    waits = 0
    stopped_early = False
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0

    for epoch in range(1, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = _run_epoch(model, train_loader, optimizer, device)
        val_metrics = _run_epoch(model, val_loader, None, device)
        if not all(torch.isfinite(torch.tensor(value)) for value in (*train_metrics.values(), *val_metrics.values())):
            raise RuntimeError(f"non-finite classifier metric at epoch {epoch}")
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - epoch_started,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        completed = epoch
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            waits = 0
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val, config, history)
        else:
            waits += 1
        log_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(
            f"mlp epoch {epoch:02d}/{config.max_epochs} "
            f"train_loss={record['train_loss']:.6f} val_loss={record['val_loss']:.6f} "
            f"train_acc={record['train_accuracy']:.4f} val_acc={record['val_accuracy']:.4f} "
            f"seconds={record['seconds']:.2f} best_epoch={best_epoch}",
            flush=True,
        )
        if waits >= config.patience:
            stopped_early = True
            print(f"mlp early_stop at epoch {epoch}", flush=True)
            break

    if best_epoch == 0:
        raise RuntimeError("classifier never stored a checkpoint")
    return ClassifierTrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val,
        training_seconds=time.perf_counter() - started,
        history=history,
        checkpoint_path=checkpoint_path,
        batch_size=config.batch_size,
        stopped_early=stopped_early,
        epochs_completed=completed,
    )
