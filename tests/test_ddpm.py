"""Scheduler, U-Net, loss, checkpoint, and CUDA checks for DDPM."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.ddpm_config import DDPMConfig
from models.diffusion.ddpm import DDPM, TRAJECTORY_LEVELS, diffusion_to_unit, unit_to_diffusion
from models.diffusion.noise_scheduler import NoiseScheduler
from models.diffusion.unet import UNet, sinusoidal_embedding
from training.ddpm_trainer import build_ddpm, fit, load_checkpoint, save_checkpoint


def _small_model(timesteps: int = 8) -> DDPM:
    unet = UNet(base_channels=8, time_dim=32)
    scheduler = NoiseScheduler(timesteps=timesteps, beta_start=1e-4, beta_end=0.02)
    return DDPM(unet, scheduler)


def test_schedule_ranges_and_monotonic_alpha_bar() -> None:
    scheduler = NoiseScheduler(timesteps=1000, beta_start=1e-4, beta_end=0.02)
    betas = scheduler.betas
    alphas = scheduler.alphas
    alpha_bar = scheduler.alpha_bar
    assert betas.shape == (1000,)
    assert torch.all(betas > 0)
    assert float(betas.min()) >= 1e-4 - 1e-8
    assert float(betas.max()) <= 0.02 + 1e-8
    assert torch.allclose(betas[0], torch.tensor(1e-4))
    assert torch.allclose(betas[-1], torch.tensor(0.02))
    assert torch.all(alphas > 0) and torch.all(alphas < 1)
    assert torch.all(alpha_bar[1:] < alpha_bar[:-1])
    assert float(scheduler.posterior_variance[0]) == 0.0
    assert torch.all(scheduler.posterior_variance[1:] > 0)


def test_forward_diffusion_and_reverse_mean_agree() -> None:
    scheduler = NoiseScheduler(timesteps=1000)
    x0 = torch.randn(8, 1, 28, 28)
    noise = torch.randn_like(x0)
    for t_index in (0, 1, 10, 100, 500, 999):
        timesteps = torch.full((x0.shape[0],), t_index)
        noisy = scheduler.add_noise(x0, timesteps, noise)
        assert noisy.shape == x0.shape
        from_epsilon = scheduler.reverse_mean(noisy, timesteps, noise)
        from_posterior = scheduler.posterior_mean(
            noisy,
            scheduler.predict_x0(noisy, timesteps, noise),
            timesteps,
        )
        # Float32 division by a small alpha_bar is largest at t = 0, about 7e-4.
        assert torch.allclose(from_epsilon, from_posterior, atol=1e-3)


def test_sinusoidal_embedding_dimension() -> None:
    timesteps = torch.arange(6)
    embedding = sinusoidal_embedding(timesteps, 256)
    assert embedding.shape == (6, 256)
    assert torch.isfinite(embedding).all()


def test_unet_output_shape_on_batch_128() -> None:
    unet = UNet()
    images = torch.randn(128, 1, 28, 28)
    timesteps = torch.randint(0, 1000, (128,))
    predicted = unet(images, timesteps)
    assert predicted.shape == (128, 1, 28, 28)
    assert torch.isfinite(predicted).all()


def test_reverse_step_loss_and_gradients() -> None:
    model = _small_model()
    images = unit_to_diffusion(torch.rand(4, 1, 28, 28))
    loss = model.noise_prediction_loss(images)
    assert torch.isfinite(loss)
    loss.backward()
    gradient = model.unet.init_conv.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    stepped = model.reverse_step(torch.randn(4, 1, 28, 28), t_index=3)
    assert stepped.shape == (4, 1, 28, 28)
    assert torch.isfinite(stepped).all()


def test_short_sample_and_unit_interval() -> None:
    model = _small_model(timesteps=4)
    frames: dict[int, torch.Tensor] = {}
    images = model.sample(2, torch.device("cpu"), frames=frames)
    assert images.shape == (2, 1, 28, 28)
    assert torch.isfinite(images).all()
    assert 4 in frames and 0 in frames
    unit = diffusion_to_unit(images)
    assert float(unit.min()) >= 0.0
    assert float(unit.max()) <= 1.0
    assert TRAJECTORY_LEVELS[0] == 1000
    assert TRAJECTORY_LEVELS[-1] == 0


def test_checkpoint_roundtrip_and_one_epoch(tmp_path) -> None:
    config = DDPMConfig(
        max_epochs=1,
        batch_size=16,
        timesteps=8,
        base_channels=8,
        time_dim=32,
        patience=5,
    )
    model = _small_model(timesteps=8)
    optimizer = torch.optim.Adam(model.unet.parameters(), lr=config.learning_rate)
    path = tmp_path / "ddpm_final.pt"
    save_checkpoint(path, model, optimizer, epoch=1, best_epoch=1, best_val_loss=0.5, waits=0, history=[{"epoch": 1}], config=config)
    restored = _small_model(timesteps=8)
    payload = load_checkpoint(path, restored)
    assert payload["epoch"] == 1
    assert payload["diffusion_timesteps"] == 8
    assert payload["seed"] == 42
    for left, right in zip(model.unet.parameters(), restored.unet.parameters()):
        assert torch.equal(left, right)

    data = torch.rand(32, 1, 28, 28)
    labels = torch.zeros(32, dtype=torch.int64)
    loader = DataLoader(TensorDataset(data, labels), batch_size=16)
    result = fit(
        _small_model(timesteps=8),
        loader,
        loader,
        config,
        torch.device("cpu"),
        tmp_path / "checkpoints",
        tmp_path / "history.json",
    )
    assert result.epochs_completed == 1
    assert result.best_checkpoint.is_file()
    assert result.final_checkpoint.is_file()
    built = build_ddpm(config)
    load_checkpoint(result.final_checkpoint, built)
    probe = built.noise_prediction_loss(unit_to_diffusion(data[:4]))
    assert torch.isfinite(probe)


def test_forward_on_cuda() -> None:
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is required for this project machine")
    device = torch.device("cuda")
    model = build_ddpm(DDPMConfig())
    model.to(device)
    images = torch.randn(8, 1, 28, 28, device=device)
    predicted = model.unet(images, torch.randint(0, 1000, (8,), device=device))
    loss = model.noise_prediction_loss(images)
    loss.backward()
    sampled = model.sample(1, device)
    unit = diffusion_to_unit(sampled)
    assert predicted.shape == (8, 1, 28, 28)
    assert predicted.device.type == "cuda"
    assert torch.isfinite(loss)
    assert sampled.shape == (1, 1, 28, 28)
    assert torch.isfinite(sampled).all()
    assert float(unit.min()) >= 0.0
    assert float(unit.max()) <= 1.0
    torch.cuda.synchronize()
