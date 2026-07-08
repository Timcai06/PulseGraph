from __future__ import annotations

from typing import Any

import torch

from app.events.run_store import RunStore, runs_dir
from app.runtime.model_loader import load_model_and_weights
from app.runtime.replay import load_probe_samples
from app.schemas import (
    CheckpointEvaluation,
    ErrorAnalysis,
    LayerHealth,
    RunDetail,
    RunInsight,
    RunReport,
)

MAX_EVALUATED_CHECKPOINTS = 6
MAX_MISCLASSIFIED_SAMPLES = 8
DEAD_LAYER_SPARSITY = 0.90
OVERFIT_GAP = 0.05
PLATEAU_TOLERANCE = 0.02


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


def _layer_health(store: RunStore, run_id: str) -> list[LayerHealth]:
    events = store.load_events(run_id)
    by_layer: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.type == "layer_snapshot" and event.layer:
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


def _evaluate_checkpoints(store: RunStore, detail: RunDetail) -> tuple[list[CheckpointEvaluation], ErrorAnalysis | None]:
    source_path = store.source_path(detail.run_id)
    entry_class = store.load_entry_class(detail.run_id)
    probe = load_probe_samples(store, detail.run_id)
    if source_path is None or entry_class is None or probe is None or not detail.checkpoints:
        return [], None

    images, labels = probe
    evaluations: list[CheckpointEvaluation] = []
    error_analysis: ErrorAnalysis | None = None
    checkpoints = detail.checkpoints[-MAX_EVALUATED_CHECKPOINTS:]

    source_root = store.run_dir(detail.run_id) / "source"
    for checkpoint in checkpoints:
        model = load_model_and_weights(source_path, entry_class, runs_dir() / checkpoint.path, source_root=source_root)
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
            error_analysis = _build_error_analysis(images, labels, predictions)

    return evaluations, error_analysis


def _build_error_analysis(images: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor) -> ErrorAnalysis:
    classes = sorted(set(labels.tolist()) | set(predictions.tolist()))
    class_index = {value: position for position, value in enumerate(classes)}
    confusion = [[0] * len(classes) for _ in classes]
    misclassified: list[dict[str, Any]] = []
    include_pixels = images.dim() == 4 and list(images.shape[1:]) == [1, 28, 28]

    for index in range(labels.numel()):
        label = int(labels[index].item())
        prediction = int(predictions[index].item())
        confusion[class_index[label]][class_index[prediction]] += 1
        if label != prediction and len(misclassified) < MAX_MISCLASSIFIED_SAMPLES:
            entry: dict[str, Any] = {"index": index, "label": label, "prediction": prediction}
            if include_pixels:
                entry["pixels"] = [round(float(v), 3) for v in images[index].flatten().tolist()]
            misclassified.append(entry)

    return ErrorAnalysis(confusion=confusion, labels=classes, misclassified=misclassified)


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
    train_losses = _metric_series(detail.metrics, "loss", phase="train") or _metric_series(detail.metrics, "loss")
    accuracies = _metric_series(detail.metrics, "accuracy")
    train_acc = _metric_series(detail.metrics, "accuracy", phase="train")
    eval_acc = _metric_series(detail.metrics, "accuracy", phase="eval")

    overfit_gap = None
    if train_acc and eval_acc:
        overfit_gap = round(train_acc[-1][1] - eval_acc[-1][1], 4)

    report = RunReport(
        run_id=detail.run_id,
        generated_for_checkpoint=detail.checkpoints[-1].step if detail.checkpoints else None,
        final_loss=train_losses[-1][1] if train_losses else None,
        best_accuracy=round(max(value for _, value in accuracies), 4) if accuracies else None,
        overfit_gap=overfit_gap,
        loss_plateau_step=_detect_plateau(train_losses),
        layer_health=_layer_health(store, detail.run_id),
    )
    report.checkpoint_evaluations, report.error_analysis = _evaluate_checkpoints(store, detail)
    report.insights = _build_insights(report, detail)
    return report
