from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from app.runtime.detection import (
    DetectionContractError,
    forward_detection,
    forward_detection_losses,
    mean_detection_iou,
    normalize_detection_batch,
    serialize_detection,
)
from app.runtime.inference_output import classification_output
from app.runtime.model_loader import forward_with_model


class TaskRuntimeError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingStepResult:
    loss: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class TelemetrySnapshot:
    layers: list[dict[str, Any]]
    metrics: dict[str, float]


@dataclass(frozen=True)
class InferenceResult:
    output: dict[str, Any]
    layers: list[dict[str, Any]]
    label: int | None = None
    prediction: int | None = None
    probabilities: list[float] | None = None


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

    def telemetry_snapshot(self, model: nn.Module, images: torch.Tensor, targets: Any) -> TelemetrySnapshot:
        return TelemetrySnapshot(layers=forward_with_model(model, images[:1])["layers"], metrics={})

    def inference(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target: Any,
        class_names: list[str] | None,
    ) -> InferenceResult:
        raise NotImplementedError

    def pack_probe_targets(self, targets: list[Any]) -> Any:
        return targets


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

    def inference(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target: Any,
        class_names: list[str] | None,
    ) -> InferenceResult:
        result = forward_with_model(model, image)
        prediction = int(result["prediction"])
        label = self.normalize_sample_target(target) if target is not None else prediction
        probabilities = list(result["probabilities"])
        return InferenceResult(
            output=classification_output(
                label=label,
                prediction=prediction,
                probabilities=probabilities,
                class_names=class_names,
            ),
            layers=result["layers"],
            label=label,
            prediction=prediction,
            probabilities=probabilities,
        )

    def pack_probe_targets(self, targets: list[Any]) -> torch.Tensor:
        return torch.tensor([self.normalize_sample_target(target) for target in targets], dtype=torch.long)


def _serialize_detection(value: Any, class_names: list[str] | None, context: str) -> dict[str, Any]:
    try:
        detection = normalize_detection_batch(value, 1, context)[0]
    except DetectionContractError as exc:
        raise TaskRuntimeError(str(exc)) from exc
    return serialize_detection(detection, class_names)


def _cpu_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_safe(item) for item in value)
    return value


class DetectionTaskRuntime(TaskRuntime):
    task = "detection"
    trainable = True

    def validate_resource_metadata(self, metadata: dict[str, Any]) -> None:
        super().validate_resource_metadata(metadata)
        class_names = metadata.get("class_names")
        if metadata.get("classes") is None and isinstance(class_names, (list, tuple)):
            metadata["classes"] = len(class_names)
        if metadata.get("classes") is None:
            raise TaskRuntimeError("Detection resources must declare 'classes' or 'class_names' in metadata().")
        metadata["classes"] = int(metadata["classes"])

    def validate_model(self, model: nn.Module, sample: torch.Tensor, metadata: dict[str, Any]) -> None:
        model.eval()
        model_input = sample if sample.dim() == 4 else sample.unsqueeze(0)
        try:
            output, _ = forward_detection(model, model_input)
            normalize_detection_batch(output, int(model_input.shape[0]), "Detection model output")
        except DetectionContractError as exc:
            raise TaskRuntimeError(str(exc)) from exc

    def normalize_sample_target(self, target: Any) -> dict[str, Any]:
        _serialize_detection(target, None, "Detection inference_sample() target")
        return target

    def serialize_sample_output(self, target: Any, class_names: list[str] | None) -> dict[str, Any]:
        return _serialize_detection(target, class_names, "Detection inference_sample() target")

    def normalize_training_batch(self, images: Any, targets: Any) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        if not isinstance(images, torch.Tensor):
            raise TaskRuntimeError("Detection train_batch() images must be a Tensor.")
        if not isinstance(targets, (list, tuple)) or len(targets) != int(images.shape[0]):
            raise TaskRuntimeError("Detection train_batch() targets must be one detection dict per image.")
        normalized: list[dict[str, Any]] = []
        for target in targets:
            _serialize_detection(target, None, "Detection train_batch() target")
            normalized.append(target)
        return images.float(), normalized

    def training_step(
        self,
        model: nn.Module,
        images: torch.Tensor,
        targets: Any,
        optimizer: torch.optim.Optimizer,
    ) -> TrainingStepResult:
        self.ensure_training_supported()
        if not isinstance(targets, list):
            raise TaskRuntimeError("Detection training targets must be a list of detection dicts.")
        try:
            losses = forward_detection_losses(model, images, targets)
        except DetectionContractError as exc:
            raise TaskRuntimeError(str(exc)) from exc
        if not isinstance(losses, dict) or not losses:
            raise TaskRuntimeError("Detection training forward must return a non-empty loss dict.")
        loss_tensors = {name: value for name, value in losses.items() if isinstance(value, torch.Tensor)}
        if not loss_tensors:
            raise TaskRuntimeError("Detection training loss dict must contain Tensor values.")
        loss = sum(loss_tensors.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return TrainingStepResult(
            loss=float(loss.detach().item()),
            metrics={name: float(value.detach().item()) for name, value in loss_tensors.items()},
        )

    def telemetry_snapshot(self, model: nn.Module, images: torch.Tensor, targets: Any) -> TelemetrySnapshot:
        try:
            output, layers = forward_detection(model, images, capture_layers=True)
            mean_iou = mean_detection_iou(output, targets, expected_size=int(images.shape[0]))
        except DetectionContractError as exc:
            raise TaskRuntimeError(str(exc)) from exc
        return TelemetrySnapshot(
            layers=layers,
            metrics={"mean_iou": mean_iou},
        )

    def inference(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target: Any,
        class_names: list[str] | None,
    ) -> InferenceResult:
        model_input = image if image.dim() == 4 else image.unsqueeze(0)
        try:
            output, layers = forward_detection(model, model_input, capture_layers=True)
        except DetectionContractError as exc:
            raise TaskRuntimeError(str(exc)) from exc
        return InferenceResult(
            output=_serialize_detection(output, class_names, "Detection model output"),
            layers=layers,
            probabilities=[],
        )

    def pack_probe_targets(self, targets: list[Any]) -> list[Any]:
        return [_cpu_safe(self.normalize_sample_target(target)) for target in targets]


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
