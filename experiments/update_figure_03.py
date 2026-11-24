"""Rebuild Figure 3 from saved checkpoints and the saved prior samples.

    .venv/bin/python experiments/update_figure_03.py

The previous figure is copied aside once. No model is trained.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloaders
from data.splits import load_split
from evaluation.classifier_analysis import latent_transpose, top_down_transpose
from models.autoencoders.vanilla_ae import VanillaAE
from models.classifier.frozen_decoder import FrozenDecoder
from models.classifier.mlp_classifier import MLPClassifier
from training.autoencoder_trainer import load_checkpoint as load_autoencoder
from training.classifier_trainer import load_checkpoint as load_classifier
from training.frozen_decoder_trainer import load_checkpoint as load_decoder
from visualization.figure_03_classifier_proof import save_figure_03

FIGURE = ROOT / "outputs" / "figures" / "figure_03_classifier_proof.png"
BACKUP = ROOT / "outputs" / "figures" / "figure_03_classifier_proof_stage5.png"


def main() -> None:
    if FIGURE.is_file() and not BACKUP.is_file():
        shutil.copy2(FIGURE, BACKUP)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with (ROOT / "configs" / "data.yaml").open(encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle)
    split = load_split(ROOT / data_config["split_path"])
    loader = build_dataloaders(
        ROOT / data_config["data_root"],
        split,
        batch_size=128,
        num_workers=0,
        seed=int(data_config["seed"]),
    )["test"]
    originals, labels = next(iter(loader))
    originals = originals[:8]
    labels = labels[:8]

    classifier = MLPClassifier()
    load_classifier(
        ROOT / "outputs" / "checkpoints" / "classifier" / "mlp_classifier_best.pt",
        classifier,
        map_location=device,
    )
    classifier.to(device)
    classifier.eval()
    decoder = FrozenDecoder()
    load_decoder(
        ROOT / "outputs" / "checkpoints" / "classifier" / "frozen_decoder_best.pt",
        decoder,
        map_location=device,
    )
    decoder.to(device)
    decoder.eval()
    autoencoder = VanillaAE(64)
    load_autoencoder(
        ROOT / "outputs" / "checkpoints" / "autoencoders" / "vanilla_ae_best.pt",
        autoencoder,
        map_location=device,
    )
    autoencoder.to(device)
    autoencoder.eval()
    prior = torch.load(
        ROOT / "outputs" / "metrics" / "mlp_prior_samples.pt",
        map_location="cpu",
        weights_only=False,
    )["images"]

    weights = classifier.get_layer_weights()
    with torch.no_grad():
        templates = top_down_transpose(weights["W1"], weights["W2"], weights["W3"]).cpu()
        features = classifier.extract_features(originals.to(device))
        projections = latent_transpose(features, weights["W1"], weights["W2"]).cpu()
        ae_images = autoencoder(originals.to(device))[0].cpu()
        decoder_images = decoder(features).cpu()
    save_figure_03(originals, templates, projections, ae_images, decoder_images, prior, labels, FIGURE)
    print(f"figure={FIGURE}", flush=True)
    print(f"backup={BACKUP}", flush=True)


if __name__ == "__main__":
    main()
