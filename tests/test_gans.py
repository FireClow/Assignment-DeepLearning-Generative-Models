"""Shape, loss, gradient, checkpoint, and CUDA checks for both GANs."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.gan_config import GanConfig
from evaluation.fid import frechet_distance, pretrained_inception_status
from models.gans.dcgan import DCGANDiscriminator, DCGANGenerator
from models.gans.vanilla_gan import VanillaDiscriminator, VanillaGenerator
from training.gan_trainer import (
    discriminator_step,
    fit_gan,
    generator_step,
    load_checkpoint,
    make_fixed_noise,
    save_checkpoint,
)


def test_vanilla_shapes_and_range() -> None:
    generator = VanillaGenerator()
    discriminator = VanillaDiscriminator()
    noise = torch.randn(128, 100)
    images = generator(noise)
    logits = discriminator(images)
    assert images.shape == (128, 1, 28, 28)
    assert logits.shape == (128, 1)
    assert float(images.detach().min()) >= -1.0
    assert float(images.detach().max()) <= 1.0
    assert torch.isfinite(images).all()
    assert torch.isfinite(logits).all()


def test_dcgan_shapes_feature_map_and_range() -> None:
    generator = DCGANGenerator()
    discriminator = DCGANDiscriminator()
    noise = torch.randn(128, 100, 1, 1)
    images = generator(noise)
    logits = discriminator(images)
    assert discriminator.feature_shape == (128, 7, 7)
    assert discriminator.feature_dim == 128 * 7 * 7
    assert images.shape == (128, 1, 28, 28)
    assert logits.shape == (128, 1)
    assert float(images.detach().min()) >= -1.0
    assert float(images.detach().max()) <= 1.0


def test_losses_and_discriminator_freeze() -> None:
    generator = VanillaGenerator()
    discriminator = VanillaDiscriminator()
    real = torch.rand(8, 1, 28, 28) * 2 - 1
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    loss_d, loss_real, loss_fake, real_logits, fake_logits = discriminator_step(
        generator, discriminator, optimizer_d, real
    )
    assert all(torch.isfinite(value) for value in (loss_d, loss_real, loss_fake))
    assert real_logits.shape == (8, 1)
    assert fake_logits.shape == (8, 1)
    before = [parameter.detach().clone() for parameter in discriminator.parameters()]
    loss_g, fake = generator_step(generator, discriminator, optimizer_g, 8, torch.device("cpu"))
    assert torch.isfinite(loss_g)
    assert fake.shape == (8, 1, 28, 28)
    assert generator.net[0].weight.grad is not None
    assert torch.isfinite(generator.net[0].weight.grad).all()
    for old, new in zip(before, discriminator.parameters()):
        assert torch.equal(old, new)


def test_checkpoint_roundtrip(tmp_path) -> None:
    generator = VanillaGenerator()
    discriminator = VanillaDiscriminator()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    noise = make_fixed_noise(4, 100, 42)
    path = tmp_path / "vanilla_gan_final.pt"
    save_checkpoint(
        path,
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        epoch=5,
        history=[{"epoch": 5}],
        config=GanConfig(),
        fixed_noise=noise,
        model_name="vanilla_gan",
    )
    restored_g = VanillaGenerator()
    restored_d = VanillaDiscriminator()
    payload = load_checkpoint(path, restored_g, restored_d)
    assert payload["epoch"] == 5
    assert payload["seed"] == 42
    assert torch.equal(payload["fixed_noise"], noise)
    for left, right in zip(generator.parameters(), restored_g.parameters()):
        assert torch.equal(left, right)


def test_fit_writes_final_checkpoint(tmp_path) -> None:
    data = torch.rand(32, 1, 28, 28)
    labels = torch.zeros(32, dtype=torch.int64)
    loader = DataLoader(TensorDataset(data, labels), batch_size=16)
    config = GanConfig(max_epochs=1, batch_size=16, grid_images=16, fixed_noise_count=16)
    result = fit_gan(
        VanillaGenerator(),
        VanillaDiscriminator(),
        loader,
        loader,
        config,
        torch.device("cpu"),
        tmp_path / "checkpoints",
        tmp_path / "history.json",
        tmp_path / "samples",
        "vanilla_gan",
        make_fixed_noise(16, 100, 42),
    )
    assert result.epochs_completed == 1
    assert (tmp_path / "checkpoints" / "vanilla_gan_final.pt").is_file()


def test_frechet_distance_cases() -> None:
    identity = np.eye(2)
    zeros = np.zeros(2)
    assert frechet_distance(zeros, identity, zeros, identity) == pytest.approx(0.0, abs=1e-5)
    shifted = np.array([3.0, 4.0])
    assert frechet_distance(zeros, identity, shifted, identity) == pytest.approx(25.0, abs=1e-5)


def test_inception_status_does_not_invent_a_score() -> None:
    status = pretrained_inception_status()
    assert isinstance(status["available"], bool)
    assert "score" not in status
    if not status["available"]:
        assert status["reason"]


def test_forward_on_cuda() -> None:
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is required for this project machine")
    device = torch.device("cuda")
    vanilla_images = VanillaGenerator().to(device)(torch.randn(8, 100, device=device))
    dcgan_images = DCGANGenerator().to(device)(torch.randn(8, 100, 1, 1, device=device))
    assert vanilla_images.device.type == "cuda"
    assert dcgan_images.shape == (8, 1, 28, 28)
    logits = DCGANDiscriminator().to(device)(dcgan_images)
    assert logits.shape == (8, 1)
    torch.cuda.synchronize()
