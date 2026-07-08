from __future__ import annotations

import importlib.util
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from app.runtime import mnist_data


class ResourceContractError(ValueError):
    pass


@dataclass
class LoadedTrainingResource:
    source_path: Path
    module: Any
    metadata: dict[str, Any]
    entry_class: str | None = None

    @property
    def name(self) -> str:
        return str(self.metadata.get("name") or self.entry_class or self.source_path.stem)

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

    @property
    def sample_source(self) -> str:
        value = self.metadata.get("sample_source") or self.metadata.get("data_source")
        return str(value) if value else "probe"

    def build_model(self) -> nn.Module:
        builder = getattr(self.module, "build_model", None)
        if callable(builder):
            model = builder()
        elif self.entry_class:
            model_class = getattr(self.module, self.entry_class, None)
            if model_class is None:
                raise ResourceContractError(f"Could not find nn.Module class '{self.entry_class}'.")
            model = model_class()
        else:
            raise ResourceContractError("Training resource must define build_model().")
        if not isinstance(model, nn.Module):
            raise ResourceContractError("build_model() must return torch.nn.Module.")
        return model

    def train_batch(self, step: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        train_batch = getattr(self.module, "train_batch", None)
        if callable(train_batch):
            images, labels = train_batch(step, batch_size)
        elif self.metadata.get("data_source") == "mnist":
            images, labels = _mnist_batch(step, batch_size)
        else:
            images, labels = _synthetic_batch(self.input_shape or [1, 28, 28], self.classes or 10, step, batch_size)
        if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise ResourceContractError("train_batch() must return (Tensor, Tensor).")
        return images.float(), labels.long()

    def inference_sample(self, index: int) -> tuple[torch.Tensor, int]:
        inference_sample = getattr(self.module, "inference_sample", None)
        if callable(inference_sample):
            image, label = inference_sample(index)
        elif self.metadata.get("data_source") == "mnist":
            image, label = _mnist_sample(index)
        else:
            image, label = _synthetic_sample(self.input_shape or [1, 28, 28], self.classes or 10, index)
        if not isinstance(image, torch.Tensor):
            raise ResourceContractError("inference_sample() must return a Tensor image.")
        return image.float(), int(label)


def _discover_entry_class(module: Any) -> str | None:
    for name, value in module.__dict__.items():
        if not isinstance(value, type):
            continue
        if value is nn.Module or not issubclass(value, nn.Module):
            continue
        if getattr(value, "__module__", None) == module.__name__:
            return name
    return None


def _probe_sample_for_model(model: nn.Module) -> torch.Tensor:
    for shape in ([1, 1, 28, 28], [1, 784], [1, 3, 32, 32], [1, 10]):
        candidate = torch.rand(*shape)
        try:
            with torch.no_grad():
                output = model(candidate)
        except Exception:
            continue
        if isinstance(output, torch.Tensor) and output.dim() == 2 and output.shape[0] == 1:
            return candidate
    raise ResourceContractError(
        "Could not infer a runnable input shape. Define inference_sample() for this model source."
    )


def _synthetic_sample(input_shape: list[int], classes: int, index: int) -> tuple[torch.Tensor, int]:
    label = int(index % max(classes, 1))
    if input_shape == [1, 28, 28]:
        image = torch.zeros(1, 1, 28, 28)
        row = 3 + (label * 2) % 20
        col = 3 + (label * 3) % 20
        image[:, :, row : row + 5, 4:24] = 0.45 + label * 0.03
        image[:, :, 4:24, col : col + 4] = 0.35 + label * 0.02
        for offset in range(20):
            y = 4 + offset if label % 2 else 23 - offset
            image[:, :, y, 4 + offset] = 0.9
        return image.clamp(0, 1), label
    values = torch.linspace(0, 1, steps=max(1, int(torch.tensor(input_shape).prod().item())))
    image = values.reshape(1, *input_shape)
    return ((image + label / max(classes, 1)) % 1).float(), label


def _synthetic_batch(input_shape: list[int], classes: int, step: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for offset in range(batch_size):
        image, label = _synthetic_sample(input_shape, classes, step + offset)
        images.append(image)
        labels.append(label)
    return torch.cat(images, dim=0), torch.tensor(labels, dtype=torch.long)


def _mnist_available() -> bool:
    return mnist_data.load_train_samples() is not None and mnist_data.load_test_samples() is not None


def _is_mnist_classifier(input_shape: list[int] | None, classes: int | None) -> bool:
    return input_shape == [1, 28, 28] and classes == 10


def _mnist_batch(step: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    data = mnist_data.load_train_samples()
    if data is None:
        raise ResourceContractError("MNIST train data is unavailable.")
    images, labels = data
    start = ((step - 1) * batch_size) % images.shape[0]
    positions = (torch.arange(batch_size) + start) % images.shape[0]
    return images[positions].float(), labels[positions].long()


def _mnist_sample(index: int) -> tuple[torch.Tensor, int]:
    data = mnist_data.load_test_samples()
    if data is None:
        raise ResourceContractError("MNIST test data is unavailable.")
    images, labels = data
    position = index % images.shape[0]
    return images[position : position + 1].float(), int(labels[position].item())


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
    entry_class = None
    if not callable(getattr(module, "build_model", None)):
        entry_class = _discover_entry_class(module)
        if entry_class is None:
            raise ResourceContractError("Training resource must define build_model() or an nn.Module subclass.")
    metadata_fn = getattr(module, "metadata", None)
    metadata = metadata_fn() if callable(metadata_fn) else {}
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ResourceContractError("metadata() must return a dict.")

    if entry_class:
        metadata.setdefault("name", entry_class)
        metadata.setdefault("source_mode", "nn-module-adapter")
    resource = LoadedTrainingResource(source_path=source_path, module=module, metadata=metadata, entry_class=entry_class)
    model = resource.build_model()
    if callable(getattr(module, "inference_sample", None)):
        sample, _ = resource.inference_sample(0)
    else:
        sample = _probe_sample_for_model(model)
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
    if entry_class and _is_mnist_classifier(resource.input_shape, resource.classes) and _mnist_available():
        resource.metadata["data_source"] = "mnist"
        resource.metadata["sample_source"] = "mnist"
    elif entry_class:
        resource.metadata.setdefault("data_source", "synthetic")
        resource.metadata.setdefault("sample_source", "synthetic")
    else:
        resource.metadata.setdefault("sample_source", "probe")
    return resource
