from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from app.runtime.model_loader import forward_with_model


class TaskRuntimeError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingStepResult:
    loss: float
    metrics: dict[str, float]


class TaskRuntime:
    task = ""
    trainable = False

    def ensure_training_supported(self) -> None:
        if not self.trainable:
            raise TaskRuntimeError(
                f"Task '{self.task}' supports resource import and preview, "
                "but training is not available in this runtime yet."
            )

    def normalize_sample_target(self, target: Any) -> Any:
        return target

    def normalize_training_batch(self, images: Any, targets: Any) -> tuple[torch.Tensor, Any]:
        if not isinstance(images, torch.Tensor):
            raise TaskRuntimeError("train_batch() images must be a Tensor.")
        return images.float(), targets

    def validate_resource_metadata(self, metadata: dict[str, Any]) -> None:
        output_schema = metadata.get("output_schema")
        if output_schema is not None and not isinstance(output_schema, dict):
            raise TaskRuntimeError("metadata()['output_schema'] must be a dict.")
        if isinstance(output_schema, dict):
            kind = output_schema.get("kind")
            if kind is not None and str(kind) != self.task:
                raise TaskRuntimeError(
                    f"metadata()['output_schema']['kind'] must match task='{self.task}'."
                )

    def validate_model(self, model: nn.Module, sample: torch.Tensor, metadata: dict[str, Any]) -> None:
        raise NotImplementedError

    def serialize_sample_output(self, target: Any, class_names: list[str] | None) -> dict[str, Any]:
        raise NotImplementedError

    def training_step(
        self,
        model: nn.Module,
        images: torch.Tensor,
        targets: Any,
        optimizer: torch.optim.Optimizer,
    ) -> TrainingStepResult:
        self.ensure_training_supported()
        raise AssertionError("unreachable")

    def capture_layers(self, model: nn.Module, images: torch.Tensor) -> list[dict[str, Any]]:
        return forward_with_model(model, images)["layers"]


class ClassificationTaskRuntime(TaskRuntime):
    task = "classification"
    trainable = True

    def normalize_sample_target(self, target: Any) -> int:
        try:
            return int(target)
        except (TypeError, ValueError) as exc:
            raise TaskRuntimeError("Classification inference_sample() target must be an integer label.") from exc

    def normalize_training_batch(self, images: Any, targets: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(images, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TaskRuntimeError("train_batch() must return (Tensor, Tensor) for classification.")
        return images.float(), targets.long()

    def validate_model(self, model: nn.Module, sample: torch.Tensor, metadata: dict[str, Any]) -> None:
        model_input = sample.unsqueeze(0) if sample.dim() in {1, 3} else sample
        with torch.no_grad():
            output = model(model_input)
        if not isinstance(output, torch.Tensor) or output.dim() != 2:
            raise TaskRuntimeError("Classification model output must have shape [batch, classes].")
        metadata.setdefault("classes", int(output.shape[1]))
        metadata["classes"] = int(metadata["classes"])
        if int(output.shape[1]) != metadata["classes"]:
            raise TaskRuntimeError("metadata()['classes'] must match the model output width.")

    def serialize_sample_output(self, target: Any, class_names: list[str] | None) -> dict[str, Any]:
        label = self.normalize_sample_target(target)
        label_name = class_names[label] if class_names is not None and 0 <= label < len(class_names) else None
        return {"kind": "classification", "label": label, "label_name": label_name}

    def training_step(
        self,
        model: nn.Module,
        images: torch.Tensor,
        targets: Any,
        optimizer: torch.optim.Optimizer,
    ) -> TrainingStepResult:
        if not isinstance(targets, torch.Tensor):
            raise TaskRuntimeError("Classification training targets must be a Tensor.")
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = float((logits.argmax(dim=1) == targets).float().mean().item())
        return TrainingStepResult(loss=float(loss.item()), metrics={"accuracy": accuracy})


def _as_detection_item(value: Any, context: str) -> dict[str, Any]:
    if isinstance(value, (list, tuple)):
        if not value:
            raise TaskRuntimeError(f"{context} must contain at least one detection item.")
        value = value[0]
    if not isinstance(value, dict):
        raise TaskRuntimeError(f"{context} must be a detection dict or a list of detection dicts.")
    if "boxes" not in value or "labels" not in value:
        raise TaskRuntimeError(f"{context} must contain 'boxes' and 'labels'.")
    return value


def _as_list(value: Any, field: str) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    raise TaskRuntimeError(f"Detection '{field}' must be a Tensor or list.")


def _serialize_detection(value: Any, class_names: list[str] | None, context: str) -> dict[str, Any]:
    item = _as_detection_item(value, context)
    boxes = _as_list(item["boxes"], "boxes")
    labels = [int(label) for label in _as_list(item["labels"], "labels")]
    scores = [float(score) for score in _as_list(item["scores"], "scores")] if "scores" in item else []
    if any(not isinstance(box, list) or len(box) != 4 for box in boxes):
        raise TaskRuntimeError("Detection 'boxes' must have shape [N, 4] in xyxy format.")
    if len(boxes) != len(labels):
        raise TaskRuntimeError("Detection 'boxes' and 'labels' must have the same length.")
    if scores and len(scores) != len(boxes):
        raise TaskRuntimeError("Detection 'scores' must match the number of boxes.")
    label_names = [
        class_names[label] if class_names is not None and 0 <= label < len(class_names) else str(label)
        for label in labels
    ]
    return {
        "kind": "detection",
        "boxes": [[float(coordinate) for coordinate in box] for box in boxes],
        "labels": labels,
        "scores": scores,
        "label_names": label_names,
    }


class DetectionTaskRuntime(TaskRuntime):
    task = "detection"

    def validate_resource_metadata(self, metadata: dict[str, Any]) -> None:
        super().validate_resource_metadata(metadata)
        class_names = metadata.get("class_names")
        if metadata.get("classes") is None and isinstance(class_names, (list, tuple)):
            metadata["classes"] = len(class_names)
        if metadata.get("classes") is None:
            raise TaskRuntimeError("Detection resources must declare 'classes' or 'class_names' in metadata().")
        metadata["classes"] = int(metadata["classes"])

    def validate_model(self, model: nn.Module, sample: torch.Tensor, metadata: dict[str, Any]) -> None:
        unbatched = sample[0] if sample.dim() == 4 and sample.shape[0] == 1 else sample
        model.eval()
        try:
            with torch.no_grad():
                output = model([unbatched])
        except Exception as list_exc:
            model_input = sample.unsqueeze(0) if sample.dim() == 3 else sample
            try:
                with torch.no_grad():
                    output = model(model_input)
            except Exception as tensor_exc:
                raise TaskRuntimeError(
                    "Detection model must accept a torchvision-style image list or a batched image Tensor. "
                    f"List input failed: {list_exc}; Tensor input failed: {tensor_exc}"
                ) from tensor_exc
        _serialize_detection(output, None, "Detection model output")

    def normalize_sample_target(self, target: Any) -> dict[str, Any]:
        _serialize_detection(target, None, "Detection inference_sample() target")
        return target

    def serialize_sample_output(self, target: Any, class_names: list[str] | None) -> dict[str, Any]:
        return _serialize_detection(target, class_names, "Detection inference_sample() target")


_RUNTIMES: dict[str, TaskRuntime] = {
    "classification": ClassificationTaskRuntime(),
    "detection": DetectionTaskRuntime(),
}


def resolve_task_runtime(task: str) -> TaskRuntime:
    normalized = str(task).strip().lower()
    runtime = _RUNTIMES.get(normalized)
    if runtime is None:
        supported = ", ".join(_RUNTIMES)
        raise TaskRuntimeError(f"Unsupported CV task '{normalized}'. Supported tasks: {supported}.")
    return runtime
