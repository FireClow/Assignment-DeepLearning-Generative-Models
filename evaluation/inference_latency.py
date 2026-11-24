"""Generator latency for a fixed batch, with warmup and CUDA synchronization."""

from __future__ import annotations

import time

import torch
from torch import nn

from training.gan_trainer import noise_for


def measure_generator_latency(
    generator: nn.Module,
    device: torch.device,
    batch_size: int = 64,
    warmup: int = 10,
    repeats: int = 30,
) -> dict:
    generator.eval()
    source = torch.randn(batch_size, generator.latent_dim, device=device)
    noise = noise_for(generator, source)
    with torch.no_grad():
        for _ in range(warmup):
            generator(noise)
            if device.type == "cuda":
                torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            generator(noise)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - started)
    mean_seconds = float(sum(samples) / len(samples))
    return {
        "device": str(device),
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": repeats,
        "mean_seconds": mean_seconds,
        "mean_milliseconds": mean_seconds * 1000.0,
    }


def measure_ddpm_latency(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 64,
    warmup: int = 1,
    repeats: int = 3,
) -> dict:
    """Wall time of one full ancestral sample, averaged over completed repeats.

    ``model.sample`` must walk every scheduler timestep. A 1,000-step DDPM is
    not comparable to a single generator forward, so this record keeps
    ``reverse_steps`` and a label of its own.
    """
    steps = int(model.scheduler.num_timesteps)
    model.unet.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model.sample(batch_size, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            model.sample(batch_size, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - started)
    mean_seconds = float(sum(samples) / len(samples))
    return {
        "device": str(device),
        "batch_size": batch_size,
        "reverse_steps": steps,
        "warmup": warmup,
        "repeats": repeats,
        "mean_seconds": mean_seconds,
        "mean_milliseconds": mean_seconds * 1000.0,
        "label": f"ddpm_{steps}_reverse_steps",
    }
