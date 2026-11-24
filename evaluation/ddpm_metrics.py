"""Noise-prediction MSE on a loader whose order does not change.

The private generator is reseeded at the start of every call, so the same
images meet the same timesteps and the same epsilon. This is the validation
protocol and the test protocol. It is not a reconstruction metric: the target
is the Gaussian noise, not the clean image.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from models.diffusion.ddpm import DDPM, unit_to_diffusion


@torch.no_grad()
def fixed_seed_noise_mse(
    model: DDPM,
    loader: DataLoader,
    device: torch.device,
    seed: int,
    split_name: str,
) -> dict:
    generator = torch.Generator()
    generator.manual_seed(seed)
    was_training = model.unet.training
    model.unet.eval()
    total = 0.0
    seen = 0
    for images, _labels in loader:
        images = unit_to_diffusion(images.to(device, non_blocking=device.type == "cuda"))
        noise = torch.randn(images.shape, generator=generator).to(device)
        timesteps = torch.randint(
            0,
            model.scheduler.num_timesteps,
            (images.shape[0],),
            generator=generator,
        ).to(device)
        loss = model.noise_prediction_loss(images, timesteps, noise)
        batch_size = int(images.shape[0])
        total += float(loss) * batch_size
        seen += batch_size
    model.unet.train(was_training)
    if seen == 0:
        raise RuntimeError(f"{split_name} loader produced zero images")
    return {
        f"{split_name}_noise_prediction_mse": total / seen,
        "images": seen,
        "seed": seed,
        "metric": "noise_prediction_mse",
    }
