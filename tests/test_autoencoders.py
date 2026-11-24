"""Shape, loss, gradient, checkpoint, and CUDA checks. No dataset download."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.autoencoder_config import AutoencoderConfig
from evaluation.reconstruction_metrics import psnr_from_mse
from models.autoencoders.sparse_ae import SparseAE, sparse_ae_loss
from models.autoencoders.vae import VariationalAE, vae_loss
from models.autoencoders.vanilla_ae import VanillaAE, mse_reconstruction
from training.autoencoder_trainer import fit, load_checkpoint, save_checkpoint


def images(batch_size: int = 128, device: str = "cpu") -> torch.Tensor:
    return torch.rand(batch_size, 1, 28, 28, device=device)


def test_vanilla_shapes() -> None:
    model = VanillaAE()
    batch = images()
    reconstruction, latent = model(batch)
    assert reconstruction.shape == batch.shape
    assert latent.shape == (128, 64)
    assert model.latent_dim == 64


def test_sparse_latent_is_non_negative() -> None:
    model = SparseAE()
    _reconstruction, latent = model(images())
    assert latent.shape == (128, 64)
    assert torch.all(latent >= 0)


def test_vae_heads_and_reparameterization() -> None:
    model = VariationalAE()
    batch = images()
    mu, logvar = model.encode(batch)
    latent = model.reparameterize(mu, logvar)
    reconstruction, forward_mu, forward_logvar, forward_z = model(batch)
    assert mu.shape == (128, 64)
    assert logvar.shape == (128, 64)
    assert latent.shape == (128, 64)
    assert forward_mu.shape == (128, 64)
    assert forward_logvar.shape == (128, 64)
    assert forward_z.shape == (128, 64)
    assert reconstruction.shape == batch.shape


def test_losses_are_finite_and_backpropagate() -> None:
    batch = images(16)
    vanilla = VanillaAE()
    reconstruction, latent = vanilla(batch)
    vanilla_loss = mse_reconstruction(reconstruction, batch)
    vanilla_loss.backward()
    assert torch.isfinite(vanilla_loss)
    assert vanilla.encoder[1].weight.grad is not None
    assert torch.isfinite(vanilla.encoder[1].weight.grad).all()

    sparse = SparseAE()
    reconstruction, latent = sparse(batch)
    total, recon, penalty = sparse_ae_loss(reconstruction, batch, latent, lambda_sparse=1e-3)
    total.backward()
    assert all(torch.isfinite(value) for value in (total, recon, penalty))
    assert sparse.encoder[1].weight.grad is not None

    vae = VariationalAE()
    reconstruction, mu, logvar, _latent = vae(batch)
    total, recon, kl = vae_loss(reconstruction, batch, mu, logvar, beta=1.0)
    total.backward()
    assert all(torch.isfinite(value) for value in (total, recon, kl))
    assert vae.fc_mu.weight.grad is not None
    assert torch.isfinite(vae.fc_mu.weight.grad).all()


def test_checkpoint_roundtrip(tmp_path) -> None:
    model = VanillaAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = AutoencoderConfig()
    path = tmp_path / "vanilla_ae_best.pt"
    save_checkpoint(path, model, optimizer, epoch=3, val_loss=0.25, config=config, model_name="vanilla_ae")
    restored = VanillaAE()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    payload = load_checkpoint(path, restored, restored_optimizer)
    assert payload["epoch"] == 3
    assert payload["seed"] == 42
    assert payload["val_loss"] == 0.25
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.equal(left, right)


def test_fit_stores_best_checkpoint(tmp_path) -> None:
    data = torch.rand(32, 1, 28, 28)
    labels = torch.zeros(32, dtype=torch.int64)
    loader = DataLoader(TensorDataset(data, labels), batch_size=16)
    config = AutoencoderConfig(max_epochs=2, patience=5, batch_size=16)
    result = fit(
        VanillaAE(),
        loader,
        loader,
        config,
        torch.device("cpu"),
        tmp_path / "tiny.pt",
        tmp_path / "history.json",
        "vanilla_ae",
    )
    assert result.best_epoch >= 1
    restored = VanillaAE()
    payload = load_checkpoint(tmp_path / "tiny.pt", restored)
    assert payload["epoch"] == result.best_epoch


def test_psnr_of_known_mse() -> None:
    assert psnr_from_mse(0.01, data_range=1.0) == pytest.approx(20.0)


def test_forward_on_cuda() -> None:
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is required for this project machine")
    device = torch.device("cuda")
    batch = images(8, device="cuda")
    for model in (VanillaAE(), SparseAE(), VariationalAE()):
        model.to(device)
        reconstruction = model(batch)[0]
        assert reconstruction.device.type == "cuda"
        assert reconstruction.shape == batch.shape
        loss = reconstruction.mean()
        loss.backward()
        torch.cuda.synchronize()
