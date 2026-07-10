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
from app.runtime.task_runtime import TaskRuntimeError, resolve_task_runtime


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
    def class_names(self) -> list[str] | None:
        value = self.metadata.get("class_names")
        return list(value) if isinstance(value, list) else None

    @property
    def task(self) -> str:
        return str(self.metadata.get("task") or "classification").strip().lower()

    @property
    def runtime(self):
        try:
            return resolve_task_runtime(self.task)
        except TaskRuntimeError as exc:
            raise ResourceContractError(str(exc)) from exc

    def ensure_training_supported(self) -> None:
        try:
            self.runtime.ensure_training_supported()
        except TaskRuntimeError as exc:
            raise ResourceContractError(str(exc)) from exc

    @property
    def dataset_spec(self) -> dict[str, Any]:
        raw = self.metadata.get("dataset_spec") or self.metadata.get("dataset")
        spec = dict(raw) if isinstance(raw, dict) else {}
        default_kind = "image_classification" if self.task == "classification" else "object_detection"
        spec.setdefault("kind", default_kind)
        spec.setdefault("name", self.name)
        spec.setdefault("source", str(self.metadata.get("data_source") or self.sample_source))
        spec.setdefault("sample_source", self.sample_source)
        if self.input_shape is not None:
            spec.setdefault("input_shape", self.input_shape)
        if self.classes is not None:
            spec.setdefault("classes", self.classes)
        return spec

    @property
    def output_schema(self) -> dict[str, Any]:
        raw = self.metadata.get("output_schema")
        schema = dict(raw) if isinstance(raw, dict) else {}
        schema.setdefault("kind", self.task)
        if self.task == "classification":
            schema.setdefault("renderer", "probability_chart")
            schema.setdefault("probabilities", True)
            if self.classes is not None:
                schema.setdefault("classes", self.classes)
            if self.class_names is not None:
                schema.setdefault("class_names", self.class_names)
        elif self.task == "detection":
            schema.setdefault("renderer", "box_overlay")
            schema.setdefault("box_format", "xyxy")
        return schema

    @property
    def metric_schema(self) -> dict[str, Any]:
        raw = self.metadata.get("metric_schema")
        schema = dict(raw) if isinstance(raw, dict) else {}
        if self.task == "classification":
            schema.setdefault("primary", "accuracy")
            schema.setdefault("monitors", ["loss", "accuracy"])
            schema.setdefault("loss", "cross_entropy")
        elif self.task == "detection":
            schema.setdefault("primary", "mean_iou")
            schema.setdefault("monitors", ["loss", "mean_iou"])
        else:
            schema.setdefault("primary", "loss")
            schema.setdefault("monitors", ["loss"])
        return schema

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

    def train_batch(self, step: int, batch_size: int) -> tuple[torch.Tensor, Any]:
        train_batch = getattr(self.module, "train_batch", None)
        if callable(train_batch):
            images, labels = train_batch(step, batch_size)
        elif self.metadata.get("data_source") == "mnist":
            images, labels = _mnist_batch(step, batch_size)
        else:
            images, labels = _synthetic_batch(self.input_shape or [1, 28, 28], self.classes or 10, step, batch_size)
        try:
            return self.runtime.normalize_training_batch(images, labels)
        except TaskRuntimeError as exc:
            raise ResourceContractError(str(exc)) from exc

    def inference_sample(self, index: int) -> tuple[torch.Tensor, Any]:
        inference_sample = getattr(self.module, "inference_sample", None)
        if callable(inference_sample):
            image, label = inference_sample(index)
        elif self.metadata.get("data_source") == "mnist":
            image, label = _mnist_sample(index)
        else:
            image, label = _synthetic_sample(self.input_shape or [1, 28, 28], self.classes or 10, index)
        if not isinstance(image, torch.Tensor):
            raise ResourceContractError("inference_sample() must return a Tensor image.")
        image_shape_from_sample(image, self.input_shape)
        try:
            target = self.runtime.normalize_sample_target(label)
        except TaskRuntimeError as exc:
            raise ResourceContractError(str(exc)) from exc
        return image.float(), target

    def sample_output(self, target: Any) -> dict[str, Any]:
        try:
            return self.runtime.serialize_sample_output(target, self.class_names)
        except TaskRuntimeError as exc:
            raise ResourceContractError(str(exc)) from exc


def _shape_numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def image_shape_from_sample(sample: torch.Tensor, declared_shape: list[int] | None = None) -> list[int]:
    """Return display shape [C,H,W] for an inference sample."""
    shape = [int(dim) for dim in sample.shape]
    declared = [int(dim) for dim in declared_shape] if declared_shape else None
    numel = int(sample.numel())

    if declared and len(declared) == 3 and _shape_numel(declared) == numel:
        image_shape = declared
    elif len(shape) == 4 and shape[0] == 1:
        image_shape = shape[1:]
    elif len(shape) == 3:
        image_shape = shape
    elif len(shape) == 2:
        image_shape = [1, shape[0], shape[1]]
    elif len(shape) == 1:
        image_shape = [1, 1, shape[0]]
    else:
        raise ResourceContractError("inference_sample() image must be reshapeable to [C,H,W].")

    if len(image_shape) != 3 or image_shape[0] not in {1, 3} or image_shape[1] <= 0 or image_shape[2] <= 0:
        raise ResourceContractError("inference_sample() image must be reshapeable to [C,H,W] with C in {1,3}.")
    if _shape_numel(image_shape) != numel:
        raise ResourceContractError("inference_sample() image shape does not match the returned tensor size.")
    return image_shape


def model_input_from_sample(sample: torch.Tensor) -> torch.Tensor:
    """Return a batched tensor for model execution without changing legacy 2D inputs."""
    if sample.dim() == 3:
        return sample.unsqueeze(0)
    if sample.dim() == 1:
        return sample.unsqueeze(0)
    return sample


def _validate_class_names(metadata: dict[str, Any], classes: int) -> None:
    raw = metadata.get("class_names")
    if raw is None:
        return
    if not isinstance(raw, (list, tuple)) or not all(isinstance(item, str) for item in raw):
        raise ResourceContractError("metadata()['class_names'] must be a list[str].")
    if len(raw) != classes:
        raise ResourceContractError("metadata()['class_names'] length must equal classes.")
    metadata["class_names"] = list(raw)


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
    import_roots = [str(source_path.parent)]
    if source_root is not None and str(source_root) not in import_roots:
        import_roots.append(str(source_root))
    for root in reversed(import_roots):
        sys.path.insert(0, root)
    try:
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise ResourceContractError("Could not load resource module.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except ResourceContractError:
        raise
    except ModuleNotFoundError as exc:
        raise ResourceContractError(
            f"The resource imports '{exc.name}', which is not part of the upload. "
            "Upload a folder or .zip that contains the package with its folder structure, "
            "or inline the dependency into the entry file."
        ) from exc
    except Exception as exc:
        raise ResourceContractError(f"Executing the resource module failed: {exc}") from exc
    finally:
        sys.modules.pop(module_name, None)
        for root in import_roots:
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
    metadata["task"] = str(metadata.get("task") or "classification").strip().lower()
    try:
        runtime = resolve_task_runtime(metadata["task"])
        runtime.validate_resource_metadata(metadata)
    except TaskRuntimeError as exc:
        raise ResourceContractError(str(exc)) from exc

    if entry_class:
        metadata.setdefault("name", entry_class)
        metadata.setdefault("source_mode", "nn-module-adapter")
    resource = LoadedTrainingResource(source_path=source_path, module=module, metadata=metadata, entry_class=entry_class)
    model = resource.build_model()
    if callable(getattr(module, "inference_sample", None)):
        sample, _ = resource.inference_sample(0)
    else:
        sample = _probe_sample_for_model(model)
    image_shape = image_shape_from_sample(sample, resource.input_shape)
    try:
        runtime.validate_model(model, sample, resource.metadata)
    except TaskRuntimeError as exc:
        raise ResourceContractError(str(exc)) from exc
    _validate_class_names(resource.metadata, int(resource.metadata["classes"]))
    resource.metadata["input_shape"] = image_shape
    if entry_class and _is_mnist_classifier(resource.input_shape, resource.classes) and _mnist_available():
        resource.metadata["data_source"] = "mnist"
        resource.metadata["sample_source"] = "mnist"
    elif entry_class:
        resource.metadata.setdefault("data_source", "synthetic")
        resource.metadata.setdefault("sample_source", "synthetic")
    else:
        resource.metadata.setdefault("sample_source", "probe")
    resource.metadata["dataset_spec"] = resource.dataset_spec
    resource.metadata["output_schema"] = resource.output_schema
    resource.metadata["metric_schema"] = resource.metric_schema
    return resource
