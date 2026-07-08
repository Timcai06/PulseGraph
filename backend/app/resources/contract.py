from __future__ import annotations

import importlib.util
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn


class ResourceContractError(ValueError):
    pass


@dataclass
class LoadedTrainingResource:
    source_path: Path
    module: Any
    metadata: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.metadata.get("name") or self.source_path.stem)

    @property
    def classes(self) -> int | None:
        value = self.metadata.get("classes")
        return int(value) if value is not None else None

    @property
    def input_shape(self) -> list[int] | None:
        value = self.metadata.get("input_shape")
        if not isinstance(value, (list, tuple)):
            return None
        return [int(dim) for dim in value]

    @property
    def batch_size(self) -> int:
        return int(self.metadata.get("batch_size") or 8)

    @property
    def learning_rate(self) -> float:
        return float(self.metadata.get("learning_rate") or 1e-3)

    def build_model(self) -> nn.Module:
        model = self.module.build_model()
        if not isinstance(model, nn.Module):
            raise ResourceContractError("build_model() must return torch.nn.Module.")
        return model

    def train_batch(self, step: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = self.module.train_batch(step, batch_size)
        if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise ResourceContractError("train_batch() must return (Tensor, Tensor).")
        return images.float(), labels.long()

    def inference_sample(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.module.inference_sample(index)
        if not isinstance(image, torch.Tensor):
            raise ResourceContractError("inference_sample() must return a Tensor image.")
        return image.float(), int(label)


def _callable(module: Any, name: str) -> Callable:
    value = getattr(module, name, None)
    if not callable(value):
        raise ResourceContractError(f"Training resource must define {name}().")
    return value


def _load_module(source_path: Path, source_root: Path | None = None) -> Any:
    module_name = f"pulsegraph_resource_{uuid.uuid4().hex[:8]}"
    root = str(source_root or source_path.parent)
    sys.path.insert(0, root)
    try:
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise ResourceContractError("Could not load resource module.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        if root in sys.path:
            sys.path.remove(root)


def load_training_resource(source_path: Path, source_root: Path | None = None) -> LoadedTrainingResource:
    module = _load_module(source_path, source_root)
    _callable(module, "build_model")
    _callable(module, "train_batch")
    _callable(module, "inference_sample")
    metadata_fn = getattr(module, "metadata", None)
    metadata = metadata_fn() if callable(metadata_fn) else {}
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ResourceContractError("metadata() must return a dict.")

    resource = LoadedTrainingResource(source_path=source_path, module=module, metadata=metadata)
    model = resource.build_model()
    sample, _ = resource.inference_sample(0)
    if sample.dim() == 1:
        sample = sample.unsqueeze(0)
    with torch.no_grad():
        output = model(sample)
    if not isinstance(output, torch.Tensor) or output.dim() != 2:
        raise ResourceContractError("build_model()(inference_sample()[0]) must return [batch, classes].")
    resource.metadata.setdefault("classes", int(output.shape[1]))
    raw_shape = metadata.get("input_shape")
    if isinstance(raw_shape, (list, tuple)):
        input_shape = [int(dim) for dim in raw_shape]
        if input_shape == list(sample.shape) and len(input_shape) > 1:
            input_shape = input_shape[1:]
    else:
        input_shape = list(sample.shape[1:]) if sample.dim() > 1 else list(sample.shape)
    resource.metadata["input_shape"] = input_shape
    return resource
