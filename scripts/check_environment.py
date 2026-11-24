"""Stage 1 environment check.

Run from the repository root:

    .venv/bin/python scripts/check_environment.py
"""

from __future__ import annotations

import sys

import matplotlib
import numpy
import pandas
import pytest
import scipy
import skimage
import torch
import torchvision
import tqdm
import yaml

REQUIRED_PYTHON = (3, 12, 16)


def main() -> None:
    version = sys.version_info[:3]
    print(f"python={sys.version.split()[0]}")
    print(f"executable={sys.executable}")
    if version != REQUIRED_PYTHON:
        raise SystemExit(f"expected Python 3.12.16, found {version}")
    if ".venv" not in sys.executable.replace("\\", "/"):
        raise SystemExit(f"interpreter is not the project virtual environment: {sys.executable}")

    print(f"torch={torch.__version__}")
    print(f"torchvision={torchvision.__version__}")
    print(f"numpy={numpy.__version__}")
    print(f"matplotlib={matplotlib.__version__}")
    print(f"tqdm={tqdm.__version__}")
    print(f"pandas={pandas.__version__}")
    print(f"scikit_image={skimage.__version__}")
    print(f"scipy={scipy.__version__}")
    print(f"pyyaml={yaml.__version__}")
    print(f"pytest={pytest.__version__}")
    print(f"cuda_compiled={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to this PyTorch build")

    properties = torch.cuda.get_device_properties(0)
    print(f"gpu_name={torch.cuda.get_device_name(0)}")
    print(f"gpu_memory_bytes={properties.total_memory}")
    print(f"gpu_memory_gb={properties.total_memory / (1024 ** 3):.2f}")

    device = torch.device("cuda")
    left = torch.randn(256, 256, device=device)
    right = torch.randn(256, 256, device=device)
    product = left @ right
    torch.cuda.synchronize()
    if tuple(product.shape) != (256, 256) or product.device.type != "cuda":
        raise SystemExit(f"CUDA matmul left the device: shape={tuple(product.shape)} device={product.device}")
    if not bool(torch.isfinite(product).all()):
        raise SystemExit("CUDA matmul produced a non-finite value")
    print(
        f"cuda_matmul=ok shape={tuple(product.shape)} "
        f"device={product.device} sum={float(product.sum()):.6f}"
    )
    print("environment_check=ok")


if __name__ == "__main__":
    main()
