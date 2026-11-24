"""Non-saturating GAN updates and checkpoint I/O.

Loss uses BCEWithLogitsLoss with reduction 'mean'. The discriminator objective
is the sum of the real-term mean and the fake-term mean. It is not divided by
two again. The generator objective is the mean BCE of fake logits against a
target of one. Both models use this definition.

Pixels enter this module in [0, 1] and are mapped once with x * 2 - 1.
Generated Tanh outputs are mapped back with (x + 1) / 2 only for images that
will be saved or scored.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from configs.gan_config import GanConfig
from evaluation.training_stability import stability_flags
from models.gans.dcgan import DCGANGenerator
from visualization.plot_utils import save_gray_grid


def unit_to_gan(images: torch.Tensor) -> torch.Tensor:
    """Map dataset pixels [0, 1] to the Tanh range [-1, 1]. Apply this once."""
    return images * 2.0 - 1.0


def gan_to_unit(images: torch.Tensor) -> torch.Tensor:
    """Map Tanh outputs back to [0, 1] for figures and metrics."""
    return ((images + 1.0) * 0.5).clamp(0.0, 1.0)


def make_fixed_noise(count: int, latent_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randn(count, latent_dim, generator=generator)


def noise_for(generator: nn.Module, source: torch.Tensor) -> torch.Tensor:
    """DCGAN receives [B, latent, 1, 1]. Vanilla GAN receives [B, latent]."""
    if isinstance(generator, DCGANGenerator):
        return source.view(source.size(0), source.size(1), 1, 1)
    return source


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def bce_logits(logits: torch.Tensor, target_value: float) -> torch.Tensor:
    target = torch.full_like(logits, target_value)
    return F.binary_cross_entropy_with_logits(logits, target)


def prediction_rate(logits: torch.Tensor, real_target: bool) -> float:
    """Fraction of rows the discriminator labels correctly.

    A logit >= 0 is a 'real' decision because sigmoid(0) = 0.5.
    """
    predicted_real = logits.detach() >= 0
    correct = predicted_real if real_target else ~predicted_real
    return float(correct.float().mean())


def discriminator_step(
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer: torch.optim.Optimizer,
    real: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update only the discriminator. Fake images are detached from G."""
    optimizer.zero_grad(set_to_none=True)
    real_logits = discriminator(real)
    source = torch.randn(real.size(0), generator.latent_dim, device=real.device)
    fake = generator(noise_for(generator, source)).detach()
    fake_logits = discriminator(fake)
    loss_real = bce_logits(real_logits, 1.0)
    loss_fake = bce_logits(fake_logits, 0.0)
    loss = loss_real + loss_fake
    loss.backward()
    optimizer.step()
    return (
        loss.detach(),
        loss_real.detach(),
        loss_fake.detach(),
        real_logits.detach(),
        fake_logits.detach(),
    )


def generator_step(
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update only the generator.

    Discriminator parameters are frozen, and its optimizer is not stepped.
    The module stays in train mode. BatchNorm must use the current fake batch
    here; eval() would score that batch with running statistics and, on DCGAN,
    remove the generator's learning signal. The fake image is not detached.
    """
    optimizer.zero_grad(set_to_none=True)
    set_requires_grad(discriminator, False)
    source = torch.randn(batch_size, generator.latent_dim, device=device)
    fake = generator(noise_for(generator, source))
    loss = bce_logits(discriminator(fake), 1.0)
    loss.backward()
    optimizer.step()
    set_requires_grad(discriminator, True)
    return loss.detach(), fake.detach()


def save_checkpoint(
    path: Path,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
    history: list[dict],
    config: GanConfig,
    fixed_noise: torch.Tensor,
    model_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_generator_state_dict": optimizer_g.state_dict(),
        "optimizer_discriminator_state_dict": optimizer_d.state_dict(),
        "epoch": epoch,
        "history": history,
        "config": config.to_dict(),
        "seed": config.seed,
        "fixed_noise": fixed_noise.detach().cpu(),
        "model_name": model_name,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    generator: nn.Module,
    discriminator: nn.Module | None = None,
    optimizer_g: torch.optim.Optimizer | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    generator.load_state_dict(payload["generator_state_dict"])
    if discriminator is not None:
        discriminator.load_state_dict(payload["discriminator_state_dict"])
    if optimizer_g is not None:
        optimizer_g.load_state_dict(payload["optimizer_generator_state_dict"])
    if optimizer_d is not None:
        optimizer_d.load_state_dict(payload["optimizer_discriminator_state_dict"])
    return payload


def generate_images(
    generator: nn.Module,
    noise: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return images in [0, 1] from a generator checkpoint's noise tensor."""
    was_training = generator.training
    generator.eval()
    source = noise.to(device)
    with torch.no_grad():
        fake = generator(noise_for(generator, source))
    generator.train(was_training)
    return gan_to_unit(fake)


@torch.no_grad()
def validation_accuracy(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Score one pass over validation images. Weights are not updated."""
    was_generator = generator.training
    was_discriminator = discriminator.training
    generator.eval()
    discriminator.eval()
    real_correct = 0.0
    fake_correct = 0.0
    seen = 0
    for real, _labels in loader:
        real = unit_to_gan(real.to(device, non_blocking=device.type == "cuda"))
        batch_size = int(real.shape[0])
        source = torch.randn(batch_size, generator.latent_dim, device=device)
        fake = generator(noise_for(generator, source))
        real_correct += prediction_rate(discriminator(real), True) * batch_size
        fake_correct += prediction_rate(discriminator(fake), False) * batch_size
        seen += batch_size
    generator.train(was_generator)
    discriminator.train(was_discriminator)
    if seen == 0:
        raise RuntimeError("the validation loader produced zero images")
    return real_correct / seen, fake_correct / seen


@dataclass
class GanTrainResult:
    model_name: str
    epochs_completed: int
    training_seconds: float
    history: list[dict]
    final_checkpoint: Path
    batch_size: int


def _weighted_add(total: float, value: float, weight: int) -> float:
    return total + value * weight


def fit_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: GanConfig,
    device: torch.device,
    checkpoint_dir: Path,
    log_path: Path,
    sample_dir: Path,
    model_name: str,
    fixed_noise: torch.Tensor,
) -> GanTrainResult:
    generator.to(device)
    discriminator.to(device)
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2)
    )
    history: list[dict] = []
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        generator.train()
        discriminator.train()
        totals = {
            "generator_loss": 0.0,
            "discriminator_loss": 0.0,
            "discriminator_real_loss": 0.0,
            "discriminator_fake_loss": 0.0,
            "discriminator_real_accuracy": 0.0,
            "discriminator_fake_accuracy": 0.0,
        }
        seen = 0
        last_fake = None
        for real, _labels in train_loader:
            real = unit_to_gan(real.to(device, non_blocking=device.type == "cuda"))
            batch_size = int(real.shape[0])
            loss_d, loss_real, loss_fake, real_logits, fake_logits = discriminator_step(
                generator, discriminator, optimizer_d, real
            )
            loss_g, fake = generator_step(
                generator, discriminator, optimizer_g, batch_size, device
            )
            if not all(torch.isfinite(value) for value in (loss_g, loss_d, loss_real, loss_fake)):
                raise RuntimeError(f"{model_name} produced a non-finite loss at epoch {epoch}")
            totals["generator_loss"] = _weighted_add(totals["generator_loss"], float(loss_g), batch_size)
            totals["discriminator_loss"] = _weighted_add(
                totals["discriminator_loss"], float(loss_d), batch_size
            )
            totals["discriminator_real_loss"] = _weighted_add(
                totals["discriminator_real_loss"], float(loss_real), batch_size
            )
            totals["discriminator_fake_loss"] = _weighted_add(
                totals["discriminator_fake_loss"], float(loss_fake), batch_size
            )
            totals["discriminator_real_accuracy"] = _weighted_add(
                totals["discriminator_real_accuracy"], prediction_rate(real_logits, True), batch_size
            )
            totals["discriminator_fake_accuracy"] = _weighted_add(
                totals["discriminator_fake_accuracy"], prediction_rate(fake_logits, False), batch_size
            )
            seen += batch_size
            last_fake = fake
        if seen == 0 or last_fake is None:
            raise RuntimeError(f"{model_name} received no training batches")
        averages = {key: value / seen for key, value in totals.items()}
        flags = stability_flags(
            averages["generator_loss"],
            averages["discriminator_loss"],
            averages["discriminator_real_accuracy"],
            averages["discriminator_fake_accuracy"],
            last_fake,
        )
        if "non_finite" in flags:
            raise RuntimeError(f"{model_name} stability check found a non-finite value")
        val_real_accuracy, val_fake_accuracy = validation_accuracy(
            generator, discriminator, val_loader, device
        )
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - epoch_started,
            **averages,
            "val_real_accuracy": val_real_accuracy,
            "val_fake_accuracy": val_fake_accuracy,
            "generated_mean": float(last_fake.mean()),
            "generated_std": float(last_fake.std()),
            "flags": flags,
        }
        history.append(record)
        log_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(
            f"{model_name} epoch {epoch:02d}/{config.max_epochs} "
            f"G={record['generator_loss']:.4f} D={record['discriminator_loss']:.4f} "
            f"D_real={record['discriminator_real_loss']:.4f} "
            f"D_fake={record['discriminator_fake_loss']:.4f} "
            f"acc_real={record['discriminator_real_accuracy']:.3f} "
            f"acc_fake={record['discriminator_fake_accuracy']:.3f} "
            f"val_acc_real={record['val_real_accuracy']:.3f} "
            f"val_acc_fake={record['val_fake_accuracy']:.3f} "
            f"seconds={record['seconds']:.2f} flags={flags}",
            flush=True,
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == config.max_epochs:
            images = generate_images(generator, fixed_noise[: config.grid_images], device)
            save_gray_grid(
                images,
                sample_dir / f"{model_name}_epoch_{epoch:03d}.png",
                f"{model_name} epoch {epoch}",
            )
            save_checkpoint(
                checkpoint_dir / f"{model_name}_epoch_{epoch:03d}.pt",
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                epoch,
                history,
                config,
                fixed_noise,
                model_name,
            )
    final_path = checkpoint_dir / f"{model_name}_final.pt"
    save_checkpoint(
        final_path,
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        history[-1]["epoch"],
        history,
        config,
        fixed_noise,
        model_name,
    )
    return GanTrainResult(
        model_name=model_name,
        epochs_completed=history[-1]["epoch"],
        training_seconds=time.perf_counter() - started,
        history=history,
        final_checkpoint=final_path,
        batch_size=config.batch_size,
    )
