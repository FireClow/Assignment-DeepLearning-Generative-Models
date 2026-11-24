"""Train the frozen-latent decoder, compare it with Vanilla AE, then sample a prior.

    .venv/bin/python experiments/run_frozen_decoder.py

Requires the classifier checkpoint and the saved ``z_MLP`` tensors. The
classifier weights are frozen. Vanilla AE is loaded and not trained.
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
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.classifier_config import ClassifierConfig, load_classifier_config
from data.dataset import PooledFashionMNIST, build_dataloaders
from data.splits import canonical_digest, load_split
from evaluation.classifier_analysis import latent_statistics, sample_gaussian_prior
from evaluation.reconstruction_metrics import evaluate_reconstruction
from models.autoencoders.vanilla_ae import VanillaAE
from models.classifier.frozen_decoder import FrozenDecoder, FrozenLatentReconstructor
from models.classifier.mlp_classifier import MLPClassifier
from training.autoencoder_trainer import load_checkpoint as load_autoencoder
from training.classifier_trainer import load_checkpoint as load_classifier
from training.frozen_decoder_trainer import fit, load_checkpoint
from visualization.figure_03_classifier_proof import (
    save_figure_03,
    save_frozen_comparison,
    save_prior_samples,
)

SPLIT_DIGEST = "963816f417b1122c50c162765938984622f50669a4401651192ff30320d55b12"
CLASSIFIER_PATH = ROOT / "outputs" / "checkpoints" / "classifier" / "mlp_classifier_best.pt"
DECODER_PATH = ROOT / "outputs" / "checkpoints" / "classifier" / "frozen_decoder_best.pt"
AE_PATH = ROOT / "outputs" / "checkpoints" / "autoencoders" / "vanilla_ae_best.pt"
LOG_PATH = ROOT / "outputs" / "logs" / "frozen_decoder_history.json"
METRICS_DIR = ROOT / "outputs" / "metrics"
CSV_PATH = METRICS_DIR / "frozen_decoder_comparison.csv"


class FeatureImageDataset(Dataset):
    """Pair saved features with images from the same unshuffled subset."""

    def __init__(self, subset: Subset, features: torch.Tensor, labels: torch.Tensor) -> None:
        if len(subset) != int(features.shape[0]) or len(subset) != int(labels.shape[0]):
            raise ValueError("feature file length does not match the split")
        self.subset = subset
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image, label = self.subset[index]
        if int(label) != int(self.labels[index]):
            raise RuntimeError(f"label mismatch at index {index}")
        return self.features[index], image, int(label)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def feature_loaders(config: ClassifierConfig, split: dict) -> dict[str, DataLoader]:
    pooled = PooledFashionMNIST(ROOT / config.data_root)
    loaders = {}
    for name in ("train", "val", "test"):
        payload = torch.load(METRICS_DIR / f"z_mlp_{name}.pt", map_location="cpu", weights_only=False)
        if not torch.equal(payload["indices"], torch.tensor(split[name], dtype=torch.long)):
            raise RuntimeError(f"{name} feature order does not match the split indices")
        dataset = FeatureImageDataset(Subset(pooled, split[name]), payload["features"], payload["labels"])
        generator = torch.Generator().manual_seed(config.seed) if name == "train" else None
        loaders[name] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=name == "train",
            generator=generator,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def train_decoder(config: ClassifierConfig, split: dict, device: torch.device):
    active = config
    for attempt in (1, 2):
        set_seed(active.seed)
        model = FrozenDecoder(latent_dim=active.latent_dim)
        loaders = feature_loaders(active, split)
        try:
            return fit(
                model,
                loaders["train"],
                loaders["val"],
                active,
                device,
                DECODER_PATH,
                LOG_PATH,
                str(CLASSIFIER_PATH),
            )
        except torch.cuda.OutOfMemoryError:
            if attempt == 2 or active.batch_size <= 64:
                raise
            torch.cuda.empty_cache()
            active = replace(active, batch_size=64)
            print("frozen decoder CUDA out of memory; retrying with batch_size=64", flush=True)
    raise RuntimeError("frozen decoder did not train")


def main() -> None:
    config = load_classifier_config(ROOT)
    split = load_split(ROOT / config.split_path)
    if canonical_digest(split) != SPLIT_DIGEST:
        raise SystemExit("refusing to train because the split digest changed")
    ae_bytes_before = AE_PATH.stat().st_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = MLPClassifier()
    load_classifier(CLASSIFIER_PATH, classifier, map_location=device)
    classifier.to(device)
    classifier.eval()
    frozen_before = [parameter.detach().clone() for parameter in classifier.parameters()]
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    result = train_decoder(config, split, device)
    for old, new in zip(frozen_before, classifier.parameters()):
        if not torch.equal(old, new):
            raise RuntimeError("classifier parameters changed while the decoder trained")

    decoder = FrozenDecoder(latent_dim=config.latent_dim)
    load_checkpoint(result.checkpoint_path, decoder, map_location=device)
    decoder.to(device)
    decoder.eval()
    image_loaders = build_dataloaders(
        ROOT / config.data_root,
        split,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    wrapper = FrozenLatentReconstructor(classifier, decoder).to(device)
    wrapper.eval()
    decoder_metrics = evaluate_reconstruction(wrapper, image_loaders["test"], device)
    autoencoder = VanillaAE(latent_dim=config.latent_dim)
    load_autoencoder(AE_PATH, autoencoder, map_location=device)
    autoencoder.to(device)
    autoencoder.eval()
    autoencoder_metrics = evaluate_reconstruction(autoencoder, image_loaders["test"], device)
    if AE_PATH.stat().st_size != ae_bytes_before:
        raise RuntimeError("Vanilla AE checkpoint size changed")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Model", "Latent Dimension", "Test MSE", "Test PSNR", "Test SSIM"])
        writer.writeheader()
        writer.writerow(
            {
                "Model": "Vanilla AE",
                "Latent Dimension": config.latent_dim,
                "Test MSE": f"{autoencoder_metrics['mse']:.8f}",
                "Test PSNR": f"{autoencoder_metrics['psnr']:.8f}",
                "Test SSIM": f"{autoencoder_metrics['ssim']:.8f}",
            }
        )
        writer.writerow(
            {
                "Model": "MLP Frozen Decoder",
                "Latent Dimension": config.latent_dim,
                "Test MSE": f"{decoder_metrics['mse']:.8f}",
                "Test PSNR": f"{decoder_metrics['psnr']:.8f}",
                "Test SSIM": f"{decoder_metrics['ssim']:.8f}",
            }
        )

    originals, labels = next(iter(image_loaders["test"]))
    originals = originals[:8].to(device)
    labels = labels[:8]
    with torch.no_grad():
        ae_images = autoencoder(originals)[0].cpu()
        decoder_images = decoder(classifier.extract_features(originals)).cpu()
    originals_cpu = originals.cpu()
    save_frozen_comparison(originals_cpu, ae_images, decoder_images, labels)

    test_features = torch.load(METRICS_DIR / "z_mlp_test.pt", map_location="cpu", weights_only=False)
    statistics = latent_statistics(test_features["features"], config.std_epsilon)
    generator = torch.Generator().manual_seed(config.seed)
    prior_latent = sample_gaussian_prior(
        statistics["mean"],
        statistics["std"],
        config.prior_samples,
        generator,
    )
    with torch.no_grad():
        prior_images = decoder(prior_latent.to(device)).cpu()
    torch.save(
        {
            "mean": statistics["mean"],
            "std": statistics["std"],
            "raw_std": statistics["raw_std"],
            "clamped_dimensions": statistics["clamped_dimensions"],
            "std_epsilon": statistics["std_epsilon"],
            "count": statistics["count"],
            "unbiased_std": True,
            "latent": prior_latent,
            "images": prior_images,
            "seed": config.seed,
        },
        METRICS_DIR / "mlp_prior_samples.pt",
    )
    save_prior_samples(prior_images)
    templates = torch.load(METRICS_DIR / "top_down_transpose.pt", map_location="cpu", weights_only=False)["images"]
    projections = torch.load(METRICS_DIR / "latent_transpose_projection.pt", map_location="cpu", weights_only=False)
    save_figure_03(
        originals_cpu,
        templates,
        projections["projections"],
        ae_images,
        decoder_images,
        prior_images,
        labels,
    )
    summary = {
        "decoder_best_epoch": result.best_epoch,
        "decoder_epochs_completed": result.epochs_completed,
        "decoder_stopped_early": result.stopped_early,
        "decoder_best_val_mse": result.best_val_loss,
        "decoder_training_seconds": result.training_seconds,
        "decoder_batch_size": result.batch_size,
        "test_images": decoder_metrics["num_images"],
        "vanilla_ae": autoencoder_metrics,
        "mlp_frozen_decoder": decoder_metrics,
        "prior_mean": statistics["mean"].tolist(),
        "prior_std": statistics["std"].tolist(),
        "prior_clamped_dimensions": statistics["clamped_dimensions"],
        "prior_std_epsilon": statistics["std_epsilon"],
        "prior_sample_pixel_mean": float(prior_images.mean()),
        "prior_sample_pixel_std": float(prior_images.std()),
        "classifier_unchanged": True,
        "vanilla_ae_retrained": False,
    }
    (METRICS_DIR / "frozen_decoder_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"frozen_decoder best_epoch={result.best_epoch} val_mse={result.best_val_loss:.6f} "
        f"test_mse={decoder_metrics['mse']:.6f} test_psnr={decoder_metrics['psnr']:.4f} "
        f"test_ssim={decoder_metrics['ssim']:.4f} "
        f"ae_mse={autoencoder_metrics['mse']:.6f} ae_psnr={autoencoder_metrics['psnr']:.4f} "
        f"ae_ssim={autoencoder_metrics['ssim']:.4f} "
        f"seconds={result.training_seconds:.2f} clamped_std={statistics['clamped_dimensions']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
