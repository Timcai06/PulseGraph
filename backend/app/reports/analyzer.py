from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from app.events.run_store import RunStore, runs_dir
from app.resources.contract import ResourceContractError, image_shape_from_sample, load_training_resource
from app.runtime.detection import (
    DetectionContractError,
    forward_detection,
    match_detection_batch,
)
from app.runtime.model_loader import load_model_and_weights
from app.runtime.replay import load_probe_samples
from app.runtime.task_runtime import TaskRuntimeError, resolve_task_runtime
from app.schemas import (
    CheckpointEvaluation,
    DetectionAnalysis,
    DetectionBoxEvidence,
    DetectionSampleEvidence,
    ErrorAnalysis,
    LayerHealth,
    RunDetail,
    RunInsight,
    RunReport,
    TaskMetricSummary,
)

MAX_EVALUATED_CHECKPOINTS = 6
MAX_MISCLASSIFIED_SAMPLES = 8
MAX_DETECTION_EVIDENCE = 4
MAX_REPORT_IMAGE_SIDE = 160
DEAD_LAYER_SPARSITY = 0.90
OVERFIT_GAP = 0.05
PLATEAU_TOLERANCE = 0.02
TASK_NEUTRAL_METRIC_KEYS = {
    "step",
    "epoch",
    "phase",
    "loss",
    "learning_rate",
    "step_time_ms",
    "samples_per_sec",
    "elapsed_sec",
}


def _metric_series(metrics: list[dict[str, Any]], key: str, phase: str | None = None) -> list[tuple[int, float]]:
    series = []
    for metric in metrics:
        if phase is not None and metric.get("phase") != phase:
            continue
        value = metric.get(key)
        if isinstance(value, (int, float)):
            series.append((int(metric.get("step", 0)), float(value)))
    return series


def _detect_plateau(losses: list[tuple[int, float]]) -> int | None:
    """First step after which loss never improves by more than PLATEAU_TOLERANCE."""
    if len(losses) < 6:
        return None
    values = [value for _, value in losses]
    for index in range(len(values) - 4):
        if min(values[index:]) > values[index] * (1 - PLATEAU_TOLERANCE):
            return losses[index][0]
    return None


def _task_name(detail: RunDetail) -> str:
    config = detail.config or {}
    return str(config.get("task") or "classification").strip().lower() or "classification"


def _metric_label(key: str) -> str:
    return key.replace("_", " ").strip().title()


def _label_name(class_names: list[str] | None, label: int) -> str | None:
    if class_names is None or label < 0 or label >= len(class_names):
        return None
    return class_names[label]


def _bounded_image_payload(image: torch.Tensor) -> tuple[dict[str, Any], float, float]:
    try:
        image_shape = image_shape_from_sample(image)
    except ResourceContractError:
        return {}, 1.0, 1.0
    if image_shape[0] not in {1, 3}:
        return {}, 1.0, 1.0

    working = image.detach().cpu().float()
    if working.dim() == 2:
        working = working.unsqueeze(0)
    if working.dim() != 3:
        return {}, 1.0, 1.0

    _, original_height, original_width = image_shape
    scale = min(1.0, MAX_REPORT_IMAGE_SIDE / max(original_height, original_width))
    rendered_height = max(1, round(original_height * scale))
    rendered_width = max(1, round(original_width * scale))
    if (rendered_height, rendered_width) != (original_height, original_width):
        working = F.interpolate(
            working.unsqueeze(0),
            size=(rendered_height, rendered_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return (
        {
            "image_shape": [image_shape[0], rendered_height, rendered_width],
            "pixels": [round(float(value), 3) for value in working.flatten().tolist()],
        },
        rendered_width / original_width,
        rendered_height / original_height,
    )


def _sample_image_payload(image: torch.Tensor) -> dict[str, Any]:
    payload, _, _ = _bounded_image_payload(image)
    return payload


def _layer_health(store: RunStore, run_id: str) -> list[LayerHealth]:
    events = store.load_events(run_id)
    by_layer: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.type != "layer_snapshot":
            continue
        if event.payload.get("mode") == "aggregate":
            expanded = store.load_layer_snapshot(run_id, event.step)
            for layer in expanded:
                layer_id = layer.get("layer_id")
                if isinstance(layer_id, str):
                    by_layer.setdefault(layer_id, []).append(layer)
            continue
        if event.layer:
            by_layer.setdefault(event.layer, []).append(event.payload)

    health: list[LayerHealth] = []
    for layer_id, snapshots in sorted(by_layer.items()):
        sparsities = [s["activation_sparsity"] for s in snapshots if isinstance(s.get("activation_sparsity"), (int, float))]
        gradients = [s["gradient_norm"] for s in snapshots if isinstance(s.get("gradient_norm"), (int, float))]
        stds = [s["weight_std"] for s in snapshots if isinstance(s.get("weight_std"), (int, float))]
        trend: str = "unknown"
        if gradients:
            last = gradients[-1]
            trend = "vanishing" if last < 1e-4 else "exploding" if last > 100 else "stable"
        health.append(
            LayerHealth(
                layer_id=layer_id,
                mean_sparsity=round(sum(sparsities) / len(sparsities), 4) if sparsities else None,
                last_gradient_norm=gradients[-1] if gradients else None,
                gradient_trend=trend,
                weight_std_drift=round(stds[-1] - stds[0], 4) if len(stds) > 1 else None,
            )
        )
    return health


def _evaluate_classification_checkpoints(store: RunStore, detail: RunDetail) -> tuple[list[CheckpointEvaluation], ErrorAnalysis | None]:
    source_path = store.source_path(detail.run_id)
    entry_class = store.load_entry_class(detail.run_id)
    probe = load_probe_samples(store, detail.run_id)
    if source_path is None or entry_class is None or probe is None or not detail.checkpoints:
        return [], None

    images, labels, _sample_source = probe
    evaluations: list[CheckpointEvaluation] = []
    error_analysis: ErrorAnalysis | None = None
    checkpoints = detail.checkpoints[-MAX_EVALUATED_CHECKPOINTS:]

    source_root = store.run_dir(detail.run_id) / "source"
    config = detail.config or {}
    class_names = config.get("class_names") if isinstance(config.get("class_names"), list) else None
    for checkpoint in checkpoints:
        model = _load_checkpoint_model(store, detail, entry_class, source_path, source_root, checkpoint.path)
        if model is None:
            continue
        try:
            with torch.no_grad():
                logits = model(images)
            predictions = logits.argmax(dim=1)
        except Exception:
            continue
        correct = int((predictions == labels).sum().item())
        evaluations.append(
            CheckpointEvaluation(
                step=checkpoint.step,
                accuracy=round(correct / labels.numel(), 4),
                sample_count=int(labels.numel()),
            )
        )
        if checkpoint is checkpoints[-1]:
            error_analysis = _build_error_analysis(images, labels, predictions, class_names=class_names)

    return evaluations, error_analysis


def _load_checkpoint_model(
    store: RunStore,
    detail: RunDetail,
    entry_class: str,
    source_path,
    source_root,
    checkpoint_path: str,
):
    checkpoint = runs_dir() / checkpoint_path
    if (detail.config or {}).get("run_kind") == "resource-training":
        try:
            resource = load_training_resource(source_path, source_root=source_root)
            model = resource.build_model()
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(state_dict, dict):
                return None
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except (ResourceContractError, RuntimeError, OSError, ValueError):
            return None
    return load_model_and_weights(source_path, entry_class, checkpoint, source_root=source_root)


def _build_error_analysis(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: list[str] | None = None,
) -> ErrorAnalysis:
    classes = sorted(set(labels.tolist()) | set(predictions.tolist()))
    class_index = {value: position for position, value in enumerate(classes)}
    confusion = [[0] * len(classes) for _ in classes]
    misclassified: list[dict[str, Any]] = []

    for index in range(labels.numel()):
        label = int(labels[index].item())
        prediction = int(predictions[index].item())
        confusion[class_index[label]][class_index[prediction]] += 1
        if label != prediction and len(misclassified) < MAX_MISCLASSIFIED_SAMPLES:
            entry: dict[str, Any] = {
                "index": index,
                "label": label,
                "label_name": _label_name(class_names, label),
                "prediction": prediction,
                "prediction_name": _label_name(class_names, prediction),
            }
            if images.dim() >= 3:
                entry.update(_sample_image_payload(images[index]))
            misclassified.append(entry)

    return ErrorAnalysis(confusion=confusion, labels=classes, class_names=class_names, misclassified=misclassified)


def build_detection_analysis(
    images: torch.Tensor,
    predictions: Any,
    targets: Any,
    class_names: list[str] | None = None,
) -> DetectionAnalysis:
    sample_count = int(images.shape[0])
    if sample_count <= 0:
        return DetectionAnalysis()

    predicted_items, target_items, matches = match_detection_batch(predictions, targets, sample_count)

    evidence: list[DetectionSampleEvidence] = []
    sample_ious: list[float] = []

    for index, (predicted, target, match) in enumerate(zip(predicted_items, target_items, matches)):
        sample_mean_iou = round(match.mean_iou, 4) if match.mean_iou is not None else None
        if match.mean_iou is not None:
            sample_ious.append(match.mean_iou)

        image_payload, scale_x, scale_y = _bounded_image_payload(images[index])

        def scaled_box(box: list[float]) -> list[float]:
            return [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y]

        predicted_evidence = [
            DetectionBoxEvidence(
                box=scaled_box(box),
                label=predicted.labels[prediction_index],
                label_name=_label_name(class_names, predicted.labels[prediction_index]) or str(predicted.labels[prediction_index]),
                score=round(predicted.scores[prediction_index], 4) if prediction_index < len(predicted.scores) else None,
                matched_iou=round(match.predicted_ious[prediction_index], 4) if match.predicted_ious[prediction_index] is not None else None,
            )
            for prediction_index, box in enumerate(predicted.boxes)
        ]
        target_evidence = [
            DetectionBoxEvidence(
                box=scaled_box(box),
                label=target.labels[target_index],
                label_name=_label_name(class_names, target.labels[target_index]) or str(target.labels[target_index]),
                matched_iou=round(match.target_ious[target_index], 4) if match.target_ious[target_index] is not None else None,
            )
            for target_index, box in enumerate(target.boxes)
        ]
        evidence.append(
            DetectionSampleEvidence(
                sample_index=index,
                image_shape=image_payload.get("image_shape", []),
                image_pixels=image_payload.get("pixels", []),
                mean_iou=sample_mean_iou,
                predicted=predicted_evidence,
                target=target_evidence,
                predicted_total=predicted.total_count,
                target_total=target.total_count,
                predicted_truncated=predicted.truncated,
                target_truncated=target.truncated,
            )
        )

    mean_iou = round(sum(sample_ious) / len(sample_ious), 4) if sample_ious else None
    return DetectionAnalysis(mean_iou=mean_iou, evaluated_samples=sample_count, evidence=evidence[:MAX_DETECTION_EVIDENCE])


def _load_structured_detection_samples(store: RunStore, run_id: str) -> tuple[torch.Tensor, list[Any], str] | None:
    path = store.samples_path(run_id)
    if path is None:
        return None
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    images = bundle.get("images")
    targets = bundle.get("targets")
    sample_source = str(bundle.get("sample_source") or "probe")
    if not isinstance(images, torch.Tensor) or not images.shape[0] or not isinstance(targets, list):
        return None
    if len(targets) < int(images.shape[0]):
        return None
    return images.float(), targets, sample_source


def _forward_detection_predictions(model, images: torch.Tensor) -> Any | None:
    try:
        outputs, _ = forward_detection(model, images)
        return outputs
    except DetectionContractError:
        return None


def _evaluate_detection_checkpoint(store: RunStore, detail: RunDetail) -> DetectionAnalysis | None:
    source_path = store.source_path(detail.run_id)
    entry_class = store.load_entry_class(detail.run_id)
    task = _task_name(detail)
    if source_path is None or entry_class is None or task != "detection" or not detail.checkpoints:
        return None
    try:
        resolve_task_runtime(task)
    except TaskRuntimeError:
        return None

    probe = _load_structured_detection_samples(store, detail.run_id)
    if probe is None:
        return None
    images, targets, _sample_source = probe

    checkpoint = detail.checkpoints[-1]
    source_root = store.run_dir(detail.run_id) / "source"
    model = _load_checkpoint_model(store, detail, entry_class, source_path, source_root, checkpoint.path)
    if model is None:
        return None
    predictions = _forward_detection_predictions(model, images)
    if predictions is None:
        return None

    class_names = (detail.config or {}).get("class_names") if isinstance((detail.config or {}).get("class_names"), list) else None
    try:
        return build_detection_analysis(images=images, predictions=predictions, targets=targets, class_names=class_names)
    except DetectionContractError:
        return None


def _build_task_metrics(detail: RunDetail, report: RunReport) -> list[TaskMetricSummary]:
    task = report.task
    summaries: list[TaskMetricSummary] = []
    seen: set[str] = set()

    def add_metric(key: str, label: str, value: float | int | str | None, unit: str | None = None) -> None:
        if key in seen or value is None:
            return
        summaries.append(TaskMetricSummary(key=key, label=label, value=value, unit=unit))
        seen.add(key)

    if task == "classification":
        if report.checkpoint_evaluations:
            final_probe = report.checkpoint_evaluations[-1]
            add_metric("final_probe_accuracy", "Final Probe Accuracy", final_probe.accuracy)
            add_metric("evaluated_samples", "Evaluated Samples", final_probe.sample_count)
        elif report.best_accuracy is not None:
            add_metric("best_accuracy", "Best Accuracy", report.best_accuracy)
    elif task == "detection" and report.detection_analysis is not None:
        add_metric("mean_iou", "Mean IoU", report.detection_analysis.mean_iou)
        add_metric("evaluated_samples", "Evaluated Samples", report.detection_analysis.evaluated_samples)

    latest_metric_values: dict[str, float] = {}
    for metric in detail.metrics:
        for key, value in metric.items():
            if key in TASK_NEUTRAL_METRIC_KEYS or not isinstance(value, (int, float)):
                continue
            latest_metric_values[key] = round(float(value), 4)
    for key, value in sorted(latest_metric_values.items()):
        add_metric(key, _metric_label(key), value)

    return summaries


def _build_insights(report: RunReport, detail: RunDetail) -> list[RunInsight]:
    insights: list[RunInsight] = []

    if report.overfit_gap is not None and report.overfit_gap > OVERFIT_GAP:
        insights.append(
            RunInsight(
                severity="warning",
                title="Overfitting detected",
                detail=f"Training accuracy exceeds evaluation accuracy by {report.overfit_gap:.3f}.",
                suggestion="Consider dropout, weight decay, data augmentation, or early stopping.",
            )
        )

    if report.loss_plateau_step is not None and detail.metrics:
        last_step = max(int(metric.get("step", 0)) for metric in detail.metrics)
        if last_step and report.loss_plateau_step < last_step * 0.5:
            insights.append(
                RunInsight(
                    severity="info",
                    title="Loss plateaued early",
                    detail=f"Loss stopped improving meaningfully around step {report.loss_plateau_step} of {last_step}.",
                    suggestion="Shorten training, add an LR schedule, or increase model capacity.",
                )
            )

    for layer in report.layer_health:
        if layer.mean_sparsity is not None and layer.mean_sparsity > DEAD_LAYER_SPARSITY:
            insights.append(
                RunInsight(
                    severity="warning",
                    title=f"Layer {layer.layer_id} is mostly inactive",
                    detail=f"Average activation sparsity is {layer.mean_sparsity:.2f} (most units output zero).",
                    suggestion="Check initialization and learning rate, or reduce the layer width.",
                )
            )
        if layer.gradient_trend in {"vanishing", "exploding"}:
            insights.append(
                RunInsight(
                    severity="critical",
                    title=f"Gradient {layer.gradient_trend} in {layer.layer_id}",
                    detail=f"Latest gradient norm is {layer.last_gradient_norm}.",
                    suggestion="Adjust learning rate, add normalization, or use gradient clipping.",
                )
            )

    if len(report.checkpoint_evaluations) > 1:
        best = max(report.checkpoint_evaluations, key=lambda item: item.accuracy or 0)
        last = report.checkpoint_evaluations[-1]
        if (best.accuracy or 0) - (last.accuracy or 0) > 0.02:
            insights.append(
                RunInsight(
                    severity="warning",
                    title="Later checkpoints regressed",
                    detail=f"Step {best.step} scores {best.accuracy} on probe samples vs {last.accuracy} at step {last.step}.",
                    suggestion=f"Deploy the step-{best.step} checkpoint instead of the final one.",
                )
            )

    if detail.source is None:
        insights.append(
            RunInsight(
                severity="info",
                title="No model source recorded",
                detail="Forward replay and checkpoint evaluation need the model source captured at training time.",
                suggestion="Train via the training/ entrypoint (or call register_source) to capture provenance.",
            )
        )

    if not insights:
        insights.append(
            RunInsight(
                severity="info",
                title="Training looks healthy",
                detail="No overfitting, dead layers, or gradient anomalies were detected in the recorded signals.",
            )
        )
    return insights


def build_run_report(store: RunStore, detail: RunDetail) -> RunReport:
    task = _task_name(detail)
    train_losses = _metric_series(detail.metrics, "loss", phase="train") or _metric_series(detail.metrics, "loss")
    accuracies = _metric_series(detail.metrics, "accuracy")
    train_acc = _metric_series(detail.metrics, "accuracy", phase="train")
    eval_acc = _metric_series(detail.metrics, "accuracy", phase="eval")

    overfit_gap = None
    if train_acc and eval_acc:
        overfit_gap = round(train_acc[-1][1] - eval_acc[-1][1], 4)

    report = RunReport(
        run_id=detail.run_id,
        task=task,
        generated_for_checkpoint=detail.checkpoints[-1].step if detail.checkpoints else None,
        final_loss=train_losses[-1][1] if train_losses else None,
        best_accuracy=round(max(value for _, value in accuracies), 4) if accuracies else None,
        overfit_gap=overfit_gap,
        loss_plateau_step=_detect_plateau(train_losses),
        layer_health=_layer_health(store, detail.run_id),
    )

    if task == "classification":
        report.checkpoint_evaluations, report.error_analysis = _evaluate_classification_checkpoints(store, detail)
    elif task == "detection":
        report.detection_analysis = _evaluate_detection_checkpoint(store, detail)

    report.task_metrics = _build_task_metrics(detail, report)
    report.insights = _build_insights(report, detail)
    return report
