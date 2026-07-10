from __future__ import annotations

import os
import struct
from functools import lru_cache
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MODEL_PATH = REPO_ROOT / "training" / "cv" / "outputs" / "mnist_mlp.pt"
DEFAULT_MNIST_RAW_DIR = REPO_ROOT / "data" / "mnist" / "MNIST" / "raw"

IDX_IMAGE_MAGIC = 2051
IDX_LABEL_MAGIC = 2049


def trained_model_path() -> Path:
    override = os.environ.get("PULSEGRAPH_MODEL_PATH")
    return Path(override) if override else DEFAULT_MODEL_PATH


def mnist_raw_dir() -> Path:
    override = os.environ.get("PULSEGRAPH_MNIST_DIR")
    return Path(override) if override else DEFAULT_MNIST_RAW_DIR


def _read_idx_images(path: Path, limit: int) -> torch.Tensor:
    with path.open("rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != IDX_IMAGE_MAGIC:
            raise ValueError(f"Unexpected IDX image magic {magic} in {path.name}")
        count = min(count, limit)
        raw = handle.read(count * rows * cols)
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return data.reshape(count, 1, rows, cols).float() / 255.0


def _read_idx_labels(path: Path, limit: int) -> torch.Tensor:
    with path.open("rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != IDX_LABEL_MAGIC:
            raise ValueError(f"Unexpected IDX label magic {magic} in {path.name}")
        count = min(count, limit)
        raw = handle.read(count)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()


@lru_cache(maxsize=1)
def load_train_samples(limit: int = 2048) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load the first `limit` MNIST train images and labels, or None when unavailable."""
    raw_dir = mnist_raw_dir()
    images_path = raw_dir / "train-images-idx3-ubyte"
    labels_path = raw_dir / "train-labels-idx1-ubyte"
    if not images_path.exists() or not labels_path.exists():
        return None
    try:
        images = _read_idx_images(images_path, limit)
        labels = _read_idx_labels(labels_path, limit)
    except (OSError, ValueError):
        return None
    if images.shape[0] != labels.shape[0]:
        return None
    return images, labels


@lru_cache(maxsize=1)
def load_test_samples(limit: int = 512) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load the first `limit` MNIST test images and labels, or None when unavailable."""
    raw_dir = mnist_raw_dir()
    images_path = raw_dir / "t10k-images-idx3-ubyte"
    labels_path = raw_dir / "t10k-labels-idx1-ubyte"
    if not images_path.exists() or not labels_path.exists():
        return None
    try:
        images = _read_idx_images(images_path, limit)
        labels = _read_idx_labels(labels_path, limit)
    except (OSError, ValueError):
        return None
    if images.shape[0] != labels.shape[0]:
        return None
    return images, labels


@lru_cache(maxsize=1)
def test_sample_indices_by_digit() -> dict[int, list[int]] | None:
    """Map each digit 0-9 to the test-set indices holding that digit."""
    data = load_test_samples()
    if data is None:
        return None
    _, labels = data
    by_digit: dict[int, list[int]] = {digit: [] for digit in range(10)}
    for position, label in enumerate(labels.tolist()):
        by_digit[label].append(position)
    return by_digit
