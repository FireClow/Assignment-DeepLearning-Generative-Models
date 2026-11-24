"""Fashion-MNIST MLP classifier: 784 -> 256 -> 64 -> 10.

``extract_features`` returns the 64-dimensional vector after the second ReLU.
The forward pass includes each Linear layer's bias. The transpose experiments
do not: they use only ``W1``, ``W2``, and ``W3`` exactly as specified, and they
do not add those bias vectors.

PyTorch stores a Linear weight as ``[out_features, in_features]``:

    W1 is [256, 784] and maps 784 -> 256
    W2 is [64, 256] and maps 256 -> 64
    W3 is [10, 64] and maps 64 -> 10
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

IMAGE_SHAPE = (1, 28, 28)
FLAT_DIM = 1 * 28 * 28


def _flat(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or tuple(images.shape[1:]) != IMAGE_SHAPE:
        raise ValueError(f"expected [B, 1, 28, 28], got {tuple(images.shape)}")
    return images.reshape(images.shape[0], FLAT_DIM)


class MLPClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(FLAT_DIM, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Latent ``z_MLP`` with shape ``[B, 64]``, after the second ReLU."""
        hidden = F.relu(self.fc1(_flat(images)))
        return F.relu(self.fc2(hidden))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Class logits with shape ``[B, num_classes]``. No softmax."""
        return self.fc3(self.extract_features(images))

    def get_layer_weights(self) -> dict[str, torch.Tensor]:
        """Detached copies of the three weight matrices, without bias."""
        return {
            "W1": self.fc1.weight.detach().clone(),
            "W2": self.fc2.weight.detach().clone(),
            "W3": self.fc3.weight.detach().clone(),
        }
