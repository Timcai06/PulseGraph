from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from app.runtime.model_loader import forward_with_layer_capture


MAX_DETECTIONS = 100


class DetectionContractError(ValueError):
    pass


@dataclass(frozen=True)
class NormalizedDetection:
    boxes: list[list[float]]
    labels: list[int]
    scores: list[float]
    total_count: int
    truncated: bool


@dataclass(frozen=True)
class DetectionMatch:
    predicted_ious: list[float | None]
    target_ious: list[float | None]
    mean_iou: float | None


def _field_length(value: Any, field: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            raise DetectionContractError(f"Detection '{field}' must have a leading item dimension.")
        return int(value.shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    raise DetectionContractError(f"Detection '{field}' must be a Tensor or list.")


def _bounded_values(value: Any, limit: int) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value[:limit].detach().cpu().tolist()
    return list(value[:limit])


def normalize_detection_item(
    value: Any,
    context: str,
    limit: int = MAX_DETECTIONS,
) -> NormalizedDetection:
    if not isinstance(value, dict):
        raise DetectionContractError(f"{context} must be a detection dict.")
    if "boxes" not in value or "labels" not in value:
        raise DetectionContractError(f"{context} must contain 'boxes' and 'labels'.")

    box_count = _field_length(value["boxes"], "boxes")
    label_count = _field_length(value["labels"], "labels")
    if box_count != label_count:
        raise DetectionContractError("Detection 'boxes' and 'labels' must have the same length.")

    score_count = _field_length(value["scores"], "scores") if "scores" in value else 0
    if "scores" in value and score_count != box_count:
        raise DetectionContractError("Detection 'scores' must match the number of boxes.")

    bounded_count = min(box_count, max(0, limit))
    raw_boxes = _bounded_values(value["boxes"], bounded_count)
    raw_labels = _bounded_values(value["labels"], bounded_count)
    raw_scores = _bounded_values(value["scores"], bounded_count) if "scores" in value else []

    boxes: list[list[float]] = []
    for box in raw_boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise DetectionContractError("Detection 'boxes' must have shape [N, 4] in xyxy format.")
        boxes.append([float(coordinate) for coordinate in box])

    return NormalizedDetection(
        boxes=boxes,
        labels=[int(label) for label in raw_labels],
        scores=[float(score) for score in raw_scores],
        total_count=box_count,
        truncated=box_count > bounded_count,
    )


def normalize_detection_batch(
    value: Any,
    expected_size: int,
    context: str,
    limit: int = MAX_DETECTIONS,
) -> list[NormalizedDetection]:
    if isinstance(value, dict):
        if expected_size != 1:
            raise DetectionContractError(
                f"{context} returned one detection dict for a batch of {expected_size}; return one dict per image."
            )
        items = [value]
    elif isinstance(value, (list, tuple)):
        if len(value) != expected_size:
            raise DetectionContractError(
                f"{context} returned {len(value)} detection items for a batch of {expected_size}."
            )
        items = list(value)
    else:
        raise DetectionContractError(f"{context} must return a detection dict or a list of detection dicts.")

    return [normalize_detection_item(item, context, limit=limit) for item in items]


def serialize_detection(
    detection: NormalizedDetection,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    label_names = [
        class_names[label] if class_names is not None and 0 <= label < len(class_names) else str(label)
        for label in detection.labels
    ]
    return {
        "kind": "detection",
        "boxes": detection.boxes,
        "labels": detection.labels,
        "scores": detection.scores,
        "label_names": label_names,
        "total_detections": detection.total_count,
        "truncated": detection.truncated,
    }


def box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_detection(predicted: NormalizedDetection, target: NormalizedDetection) -> DetectionMatch:
    predicted_ious: list[float | None] = [None] * len(predicted.boxes)
    target_ious: list[float | None] = [None] * len(target.boxes)
    used_predictions: set[int] = set()
    target_scores: list[float] = []

    for target_index, target_box in enumerate(target.boxes):
        target_label = target.labels[target_index]
        best_prediction: int | None = None
        best_iou = 0.0
        for prediction_index, prediction_box in enumerate(predicted.boxes):
            if prediction_index in used_predictions or predicted.labels[prediction_index] != target_label:
                continue
            iou = box_iou(prediction_box, target_box)
            if iou > best_iou:
                best_iou = iou
                best_prediction = prediction_index
        if best_prediction is None:
            target_scores.append(0.0)
            continue
        used_predictions.add(best_prediction)
        predicted_ious[best_prediction] = best_iou
        target_ious[target_index] = best_iou
        target_scores.append(best_iou)

    mean_iou = sum(target_scores) / len(target_scores) if target_scores else None
    return DetectionMatch(predicted_ious=predicted_ious, target_ious=target_ious, mean_iou=mean_iou)


def match_detection_batch(
    predictions: Any,
    targets: Any,
    expected_size: int,
) -> tuple[list[NormalizedDetection], list[NormalizedDetection], list[DetectionMatch]]:
    predicted_items = normalize_detection_batch(predictions, expected_size, "Detection model output")
    target_items = normalize_detection_batch(targets, expected_size, "Detection targets")
    matches = [match_detection(predicted, target) for predicted, target in zip(predicted_items, target_items)]
    return predicted_items, target_items, matches


def mean_detection_iou(predictions: Any, targets: Any, expected_size: int) -> float:
    _, _, matches = match_detection_batch(predictions, targets, expected_size)
    scores = [match.mean_iou for match in matches if match.mean_iou is not None]
    return sum(scores) / len(scores) if scores else 0.0


def forward_detection(
    model: nn.Module,
    images: torch.Tensor,
    capture_layers: bool = False,
) -> tuple[Any, list[dict[str, Any]]]:
    if not isinstance(images, torch.Tensor) or images.dim() < 4:
        raise DetectionContractError("Detection inference images must be a batched Tensor.")

    image_list = [image for image in images]
    failures: list[str] = []
    for model_input, style in ((image_list, "image list"), (images, "batched Tensor")):
        try:
            if capture_layers:
                return forward_with_layer_capture(model, model_input)
            with torch.no_grad():
                return model(model_input), []
        except Exception as exc:
            failures.append(f"{style}: {exc}")
    raise DetectionContractError("Detection forward failed for both supported input styles. " + "; ".join(failures))


def forward_detection_losses(model: nn.Module, images: torch.Tensor, targets: list[dict[str, Any]]) -> Any:
    failures: list[str] = []
    image_list = [image for image in images]
    for model_input, style in ((image_list, "image list"), (images, "batched Tensor")):
        try:
            return model(model_input, targets)
        except Exception as exc:
            failures.append(f"{style}: {exc}")
    raise DetectionContractError("Detection training forward failed for both supported input styles. " + "; ".join(failures))
