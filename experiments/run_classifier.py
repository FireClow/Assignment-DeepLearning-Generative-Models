"""Train the MLP, then run the two transpose experiments.

    .venv/bin/python experiments/run_classifier.py

The test split is scored only after the validation checkpoint is chosen.
Feature files keep the unshuffled split order. This script then calls the
frozen-decoder experiment.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.classifier_config import ClassifierConfig, load_classifier_config
from data.dataset import PooledFashionMNIST, build_dataloaders
from data.splits import canonical_digest, load_split
from evaluation.classifier_analysis import classification_scores, latent_transpose, top_down_transpose
from experiments.run_frozen_decoder import main as run_frozen_decoder
from models.classifier.mlp_classifier import MLPClassifier
from training.classifier_trainer import ClassifierTrainResult, fit, load_checkpoint
from visualization.classifier_training_curves import save_classifier_curves
from visualization.figure_03_classifier_proof import save_latent_transpose, save_weight_transpose
from visualization.plot_utils import FASHION_CLASSES

SPLIT_DIGEST = "963816f417b1122c50c162765938984622f50669a4401651192ff30320d55b12"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "classifier"
LOG_PATH = ROOT / "outputs" / "logs" / "classifier_history.json"
METRICS_DIR = ROOT / "outputs" / "metrics"
CHECKPOINT_PATH = CHECKPOINT_DIR / "mlp_classifier_best.pt"
ACCURACY_TARGET = 0.88


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ordered_loader(root: Path, indices: list[int], batch_size: int, num_workers: int) -> DataLoader:
    subset = Subset(PooledFashionMNIST(root), indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def extract_features(model: MLPClassifier, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    features = []
    labels = []
    for images, batch_labels in loader:
        features.append(model.extract_features(images.to(device, non_blocking=device.type == "cuda")).cpu())
        labels.append(batch_labels)
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def collect_predictions(model: MLPClassifier, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    predicted = []
    labels = []
    for images, batch_labels in loader:
        logits = model(images.to(device, non_blocking=device.type == "cuda"))
        predicted.append(logits.argmax(dim=1).cpu())
        labels.append(batch_labels)
    return torch.cat(predicted), torch.cat(labels)


def train_classifier(config: ClassifierConfig, split: dict, device: torch.device) -> ClassifierTrainResult:
    active = config
    for attempt in (1, 2):
        set_seed(active.seed)
        model = MLPClassifier(num_classes=active.num_classes)
        loaders = build_dataloaders(
            ROOT / active.data_root,
            split,
            batch_size=active.batch_size,
            num_workers=active.num_workers,
            seed=active.seed,
        )
        try:
            return fit(
                model,
                loaders["train"],
                loaders["val"],
                active,
                device,
                CHECKPOINT_PATH,
                LOG_PATH,
            )
        except torch.cuda.OutOfMemoryError:
            if attempt == 2 or active.batch_size <= 64:
                raise
            torch.cuda.empty_cache()
            active = replace(active, batch_size=64)
            print("mlp CUDA out of memory; retrying with batch_size=64", flush=True)
    raise RuntimeError("classifier did not train")


def main() -> None:
    config = load_classifier_config(ROOT)
    if config.seed != 42 or config.batch_size != 128 or config.learning_rate != 1e-3:
        raise SystemExit(f"locked classifier config changed: {config}")
    split = load_split(ROOT / config.split_path)
    digest = canonical_digest(split)
    if digest != SPLIT_DIGEST:
        raise SystemExit("refusing to train because the split digest changed")
    if (len(split["train"]), len(split["val"]), len(split["test"])) != (56000, 7000, 7000):
        raise SystemExit("refusing to train on a split other than 56000/7000/7000")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    result = train_classifier(config, split, device)
    model = MLPClassifier(num_classes=config.num_classes)
    load_checkpoint(result.checkpoint_path, model, map_location=device)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    loaders = {
        name: ordered_loader(ROOT / config.data_root, split[name], result.batch_size, config.num_workers)
        for name in ("train", "val", "test")
    }
    split_predictions = {}
    for name in ("train", "val", "test"):
        predicted, labels = collect_predictions(model, loaders[name], device)
        split_predictions[name] = classification_scores(predicted, labels, config.num_classes)
    test_accuracy = float(split_predictions["test"]["accuracy"])
    best_record = result.history[result.best_epoch - 1]
    metrics = {
        "model": "mlp_classifier",
        "architecture": "784 -> 256 -> 64 -> 10",
        "epochs_completed": result.epochs_completed,
        "best_epoch": result.best_epoch,
        "stopped_early": result.stopped_early,
        "best_validation_loss": result.best_val_loss,
        "best_epoch_train_loss": best_record["train_loss"],
        "best_epoch_train_accuracy": best_record["train_accuracy"],
        "best_epoch_val_loss": best_record["val_loss"],
        "best_epoch_val_accuracy": best_record["val_accuracy"],
        "checkpoint_train_accuracy": split_predictions["train"]["accuracy"],
        "checkpoint_val_accuracy": split_predictions["val"]["accuracy"],
        "test_accuracy": test_accuracy,
        "test_macro_precision": split_predictions["test"]["macro_precision"],
        "test_macro_recall": split_predictions["test"]["macro_recall"],
        "test_macro_f1": split_predictions["test"]["macro_f1"],
        "test_per_class": {
            FASHION_CLASSES[index]: {
                "precision": split_predictions["test"]["per_class_precision"][index],
                "recall": split_predictions["test"]["per_class_recall"][index],
                "f1": split_predictions["test"]["per_class_f1"][index],
            }
            for index in range(config.num_classes)
        },
        "accuracy_target": ACCURACY_TARGET,
        "target_met": test_accuracy >= ACCURACY_TARGET,
        "training_seconds": result.training_seconds,
        "batch_size": result.batch_size,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "split_digest": digest,
        "transpose_uses_bias": False,
        "checkpoint": str(result.checkpoint_path),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "classifier_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    save_classifier_curves(result.history)
    print(
        f"mlp best_epoch={result.best_epoch} "
        f"train_acc={split_predictions['train']['accuracy']:.4f} "
        f"val_acc={split_predictions['val']['accuracy']:.4f} "
        f"test_acc={test_accuracy:.4f} "
        f"macro_f1={split_predictions['test']['macro_f1']:.4f} "
        f"seconds={result.training_seconds:.2f}",
        flush=True,
    )
    if test_accuracy < ACCURACY_TARGET:
        raise SystemExit(f"test accuracy {test_accuracy:.4f} is below {ACCURACY_TARGET:.2f}")

    for name in ("train", "val", "test"):
        features, labels = extract_features(model, loaders[name], device)
        if not torch.equal(labels, torch.cat([batch_labels for _, batch_labels in loaders[name]])):
            raise RuntimeError(f"{name} feature labels do not match a second pass")
        torch.save(
            {
                "features": features,
                "labels": labels,
                "indices": torch.tensor(split[name], dtype=torch.long),
            },
            METRICS_DIR / f"z_mlp_{name}.pt",
        )
        print(f"features {name} {tuple(features.shape)}", flush=True)

    weights = model.get_layer_weights()
    templates = top_down_transpose(weights["W1"], weights["W2"], weights["W3"]).cpu()
    originals, original_labels = next(iter(loaders["test"]))
    originals = originals[:8]
    original_labels = original_labels[:8]
    projections = latent_transpose(model.extract_features(originals.to(device)), weights["W1"], weights["W2"]).cpu()
    torch.save({"images": templates, "class_names": list(FASHION_CLASSES), "uses_bias": False}, METRICS_DIR / "top_down_transpose.pt")
    torch.save(
        {"originals": originals, "projections": projections, "labels": original_labels, "uses_bias": False},
        METRICS_DIR / "latent_transpose_projection.pt",
    )
    save_weight_transpose(templates)
    save_latent_transpose(originals, projections, original_labels)
    run_frozen_decoder()


if __name__ == "__main__":
    main()
