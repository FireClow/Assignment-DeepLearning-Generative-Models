"""MSE, PSNR, and SSIM on a full loader.

Each image contributes equally. PSNR uses data_range 1.0:

    PSNR = 10 * log10(1 / MSE_image)

SSIM uses scikit-image with data_range 1.0 and the default 7x7 window, which
fits a 28x28 grayscale image. The reported numbers are means over every image
in the loader, not over a single batch.
"""

from __future__ import annotations

import math

import torch
from skimage.metrics import structural_similarity
from torch import nn
from torch.utils.data import DataLoader


def psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    if mse < 0:
        raise ValueError(f"MSE cannot be negative, got {mse}")
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((data_range * data_range) / mse)


def reconstruct(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Every stage-2 forward returns the reconstruction as its first value."""
    return model(images)[0]


def evaluate_reconstruction(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    mse_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    seen = 0
    with torch.no_grad():
        for images, _labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            reconstruction = reconstruct(model, images).clamp(0.0, 1.0)
            per_image_mse = (reconstruction - images).pow(2).flatten(1).mean(dim=1)
            original_np = images.detach().cpu().numpy()
            restored_np = reconstruction.detach().cpu().numpy()
            for index in range(images.shape[0]):
                mse_value = float(per_image_mse[index].item())
                ssim_value = float(
                    structural_similarity(
                        original_np[index, 0],
                        restored_np[index, 0],
                        data_range=1.0,
                    )
                )
                psnr_value = psnr_from_mse(mse_value, data_range=1.0)
                if not all(math.isfinite(value) for value in (mse_value, psnr_value, ssim_value)):
                    raise RuntimeError("reconstruction metric is not finite")
                mse_sum += mse_value
                psnr_sum += psnr_value
                ssim_sum += ssim_value
                seen += 1
    if seen == 0:
        raise RuntimeError("the evaluation loader produced zero images")
    return {
        "mse": mse_sum / seen,
        "psnr": psnr_sum / seen,
        "ssim": ssim_sum / seen,
        "num_images": float(seen),
    }
