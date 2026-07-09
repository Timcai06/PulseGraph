"""Self-contained RGB pattern classification resource for PulseGraph verification.

Upload this file directly, or zip the examples/resources folder. It uses
deterministic 32x32 RGB pattern images so the workflow does not depend on
network access or pretend to use real-world CIFAR photos.
"""

from __future__ import annotations

import torch
from torch import nn

CLASS_NAMES = [
    "pattern_00",
    "pattern_01",
    "pattern_02",
    "pattern_03",
    "pattern_04",
    "pattern_05",
    "pattern_06",
    "pattern_07",
    "pattern_08",
    "pattern_09",
]


class CifarRgbExampleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, len(CLASS_NAMES)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


def metadata() -> dict:
    return {
        "name": "rgb_pattern_example",
        "classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "input_shape": [3, 32, 32],
        "batch_size": 16,
        "learning_rate": 1e-3,
        "data_source": "synthetic-rgb-patterns",
        "sample_source": "probe",
    }


def build_model() -> nn.Module:
    return CifarRgbExampleNet()


def _image_for_label(label: int) -> torch.Tensor:
    image = torch.zeros(3, 32, 32)
    channel = label % 3
    image[channel].fill_(0.25 + 0.05 * (label % 6))
    row = 3 + (label * 3) % 22
    col = 3 + (label * 5) % 22
    image[:, row : row + 6, :] += 0.16
    image[:, :, col : col + 5] += 0.12
    image[(channel + 1) % 3, 7:25, 7:25] += 0.14
    return image.clamp(0, 1)


def train_batch(step: int, batch_size: int):
    labels = torch.tensor([(step - 1 + offset) % len(CLASS_NAMES) for offset in range(batch_size)], dtype=torch.long)
    images = torch.stack([_image_for_label(int(label)) for label in labels], dim=0)
    return images, labels


def inference_sample(index: int):
    label = index % len(CLASS_NAMES)
    return _image_for_label(label), label
