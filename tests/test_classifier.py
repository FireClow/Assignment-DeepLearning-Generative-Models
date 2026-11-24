"""Shape, transpose, frozen-decoder, checkpoint, and CUDA checks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from configs.classifier_config import ClassifierConfig
from evaluation.classifier_analysis import latent_transpose, sample_gaussian_prior, top_down_transpose
from models.classifier.frozen_decoder import FrozenDecoder
from models.classifier.mlp_classifier import MLPClassifier
from training.classifier_trainer import fit as fit_classifier
from training.classifier_trainer import load_checkpoint as load_classifier
from training.classifier_trainer import save_checkpoint as save_classifier
from training.frozen_decoder_trainer import fit as fit_decoder
from training.frozen_decoder_trainer import load_checkpoint as load_decoder
from training.frozen_decoder_trainer import save_checkpoint as save_decoder


def test_mlp_shapes_are_finite() -> None:
    model = MLPClassifier()
    images = torch.rand(128, 1, 28, 28)
    logits = model(images)
    features = model.extract_features(images)
    weights = model.get_layer_weights()
    assert logits.shape == (128, 10)
    assert features.shape == (128, 64)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(features).all()
    assert torch.all(features >= 0)
    assert tuple(weights["W1"].shape) == (256, 784)
    assert tuple(weights["W2"].shape) == (64, 256)
    assert tuple(weights["W3"].shape) == (10, 64)


def test_transpose_projections() -> None:
    model = MLPClassifier()
    weights = model.get_layer_weights()
    templates = top_down_transpose(weights["W1"], weights["W2"], weights["W3"])
    latent = torch.relu(torch.randn(8, 64))
    projected = latent_transpose(latent, weights["W1"], weights["W2"])
    assert templates.shape == (10, 1, 28, 28)
    assert projected.shape == (8, 1, 28, 28)
    assert float(templates.min()) >= 0.0 and float(templates.max()) <= 1.0
    assert torch.isfinite(projected).all()
    manual = torch.sigmoid(torch.relu(latent @ weights["W2"]) @ weights["W1"])
    assert torch.allclose(projected.flatten(1), manual)


def test_decoder_ignores_classifier_parameters() -> None:
    classifier = MLPClassifier()
    decoder = FrozenDecoder()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    before = [parameter.detach().clone() for parameter in classifier.parameters()]
    images = torch.rand(8, 1, 28, 28)
    features = classifier.extract_features(images)
    reconstruction = decoder(features)
    loss = F.mse_loss(reconstruction, images)
    assert reconstruction.shape == (8, 1, 28, 28)
    assert torch.isfinite(loss)
    loss.backward()
    assert decoder.net[0].weight.grad is not None
    assert torch.isfinite(decoder.net[0].weight.grad).all()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    optimizer.step()
    for old, new in zip(before, classifier.parameters()):
        assert torch.equal(old, new)
    assert all(parameter.grad is None for parameter in classifier.parameters())


def test_prior_is_finite() -> None:
    mean = torch.zeros(64)
    std = torch.ones(64)
    generator = torch.Generator().manual_seed(42)
    samples = sample_gaussian_prior(mean, std, 16, generator)
    assert samples.shape == (16, 64)
    assert torch.isfinite(samples).all()


def test_checkpoints_roundtrip(tmp_path) -> None:
    config = ClassifierConfig(max_epochs=1, decoder_max_epochs=1, batch_size=16, patience=7, decoder_patience=7)
    classifier = MLPClassifier()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    classifier_path = tmp_path / "mlp_classifier_best.pt"
    save_classifier(classifier_path, classifier, optimizer, 1, 0.5, config, [{"epoch": 1}])
    restored = MLPClassifier()
    payload = load_classifier(classifier_path, restored)
    assert payload["seed"] == 42
    assert payload["best_epoch"] == 1
    for left, right in zip(classifier.parameters(), restored.parameters()):
        assert torch.equal(left, right)

    decoder = FrozenDecoder()
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    decoder_path = tmp_path / "frozen_decoder_best.pt"
    save_decoder(decoder_path, decoder, decoder_optimizer, 1, 0.2, config, [{"epoch": 1}], str(classifier_path))
    restored_decoder = FrozenDecoder()
    load_decoder(decoder_path, restored_decoder)
    for left, right in zip(decoder.parameters(), restored_decoder.parameters()):
        assert torch.equal(left, right)

    images = torch.rand(32, 1, 28, 28)
    labels = torch.randint(0, 10, (32,))
    image_loader = DataLoader(TensorDataset(images, labels), batch_size=16)
    result = fit_classifier(
        MLPClassifier(),
        image_loader,
        image_loader,
        config,
        torch.device("cpu"),
        tmp_path / "trained_classifier.pt",
        tmp_path / "classifier_history.json",
    )
    assert result.epochs_completed == 1
    assert result.checkpoint_path.is_file()

    features = torch.relu(torch.randn(32, 64))
    feature_loader = DataLoader(TensorDataset(features, images, labels), batch_size=16)
    decoder_result = fit_decoder(
        FrozenDecoder(),
        feature_loader,
        feature_loader,
        config,
        torch.device("cpu"),
        tmp_path / "trained_decoder.pt",
        tmp_path / "decoder_history.json",
        str(classifier_path),
    )
    assert decoder_result.checkpoint_path.is_file()


def test_forward_on_cuda() -> None:
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is required for this project machine")
    device = torch.device("cuda")
    classifier = MLPClassifier().to(device)
    decoder = FrozenDecoder().to(device)
    images = torch.rand(128, 1, 28, 28, device=device)
    logits = classifier(images)
    features = classifier.extract_features(images)
    reconstruction = decoder(features)
    weights = classifier.get_layer_weights()
    templates = top_down_transpose(weights["W1"], weights["W2"], weights["W3"])
    assert logits.shape == (128, 10)
    assert features.shape == (128, 64)
    assert reconstruction.shape == (128, 1, 28, 28)
    assert templates.shape == (10, 1, 28, 28)
    assert logits.device.type == "cuda"
    torch.cuda.synchronize()
