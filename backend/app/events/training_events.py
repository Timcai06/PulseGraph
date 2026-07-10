from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from app.events.run_registry import RunRegistry
from app.events.run_store import RunStore
from app.schemas import RunEvent

RunEventType = Literal[
    "run_status",
    "metric",
    "layer_snapshot",
    "infra",
    "checkpoint",
    "animation",
    "graph",
    "source_registered",
    "config_registered",
    "graph_registered",
    "run_complete",
]
RunEventSource = Literal["training", "runtime_hook", "checkpoint", "infra", "plugin", "animation"]


def build_run_event(
    run_id: str,
    event_type: RunEventType,
    source: RunEventSource,
    step: int,
    payload: dict[str, Any],
    layer: str | None = None,
) -> RunEvent:
    return RunEvent(
        event_id=str(uuid.uuid4()),
        ts_ns=time.time_ns(),
        source=source,
        type=event_type,
        run_id=run_id,
        step=step,
        layer=layer,
        payload=payload,
    )


def save_run_config(run_store: RunStore, run_id: str, **updates: Any) -> dict[str, Any]:
    return run_store.update_config(run_id, updates)


def publish_run_status(
    run_registry: RunRegistry,
    run_id: str,
    phase: str,
    message: str,
    *,
    step: int = 0,
    total_steps: int | None = None,
    elapsed_sec: float | None = None,
    eta_sec: float | None = None,
    progress: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {"phase": phase, "message": message}
    if total_steps is not None:
        payload["total_steps"] = total_steps
    if step:
        payload["step"] = step
    if elapsed_sec is not None:
        payload["elapsed_sec"] = round(elapsed_sec, 2)
    if eta_sec is not None:
        payload["eta_sec"] = round(max(eta_sec, 0.0), 2)
    if progress is not None:
        payload["progress"] = round(min(max(progress, 0.0), 1.0), 4)
    if extra:
        payload.update(extra)
    run_registry.publish(run_id, [build_run_event(run_id, "run_status", "training", step, payload)])


def _numeric_mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate_layer_snapshot(
    run_store: RunStore,
    run_id: str,
    step: int,
    layers: list[dict[str, Any]],
    *,
    max_live_layer_samples: int,
) -> RunEvent:
    detail_path = run_store.save_layer_snapshot(run_id, step, layers)
    sample_candidates = sorted(
        layers,
        key=lambda layer: float(layer.get("gradient_norm") or 0.0),
        reverse=True,
    )
    sampled_layers = sample_candidates[:max_live_layer_samples]
    activation_means = [float(layer["activation_mean"]) for layer in layers if isinstance(layer.get("activation_mean"), (int, float))]
    sparsities = [float(layer["activation_sparsity"]) for layer in layers if isinstance(layer.get("activation_sparsity"), (int, float))]
    gradients = [float(layer["gradient_norm"]) for layer in layers if isinstance(layer.get("gradient_norm"), (int, float))]
    weight_stds = [float(layer["weight_std"]) for layer in layers if isinstance(layer.get("weight_std"), (int, float))]
    payload = {
        "mode": "aggregate",
        "reduced": len(layers) > len(sampled_layers),
        "layer_count": len(layers),
        "sampled_layer_count": len(sampled_layers),
        "detail_path": detail_path,
        "layers": sampled_layers,
        "activation_mean": _numeric_mean(activation_means),
        "activation_sparsity": _numeric_mean(sparsities),
        "gradient_norm": _numeric_mean(gradients),
        "gradient_norm_max": round(max(gradients), 4) if gradients else None,
        "weight_std": _numeric_mean(weight_stds),
    }
    return build_run_event(run_id, "layer_snapshot", "runtime_hook", step, payload, layer="__aggregate__")


def publish_training_progress(
    run_registry: RunRegistry,
    run_store: RunStore,
    run_id: str,
    step: int,
    steps: int,
    learning_rate: float,
    step_result: Any,
    step_time_ms: float,
    elapsed_sec: float,
    layers: list[dict[str, Any]] | None,
    telemetry_metrics: dict[str, float] | None,
    batch_size: int,
    *,
    max_live_layer_samples: int,
) -> None:
    progress = step / max(steps, 1)
    eta_sec = (elapsed_sec / step) * max(steps - step, 0) if step else None
    status_extra = {
        "loss": round(step_result.loss, 4),
        "learning_rate": learning_rate,
        "step_time_ms": round(step_time_ms, 2),
    }
    for key, value in step_result.metrics.items():
        status_extra[key] = round(float(value), 4)
    if telemetry_metrics:
        for key, value in telemetry_metrics.items():
            status_extra[key] = round(float(value), 4)
    publish_run_status(
        run_registry,
        run_id,
        "training",
        f"Training step {step}/{steps}",
        step=step,
        total_steps=steps,
        elapsed_sec=elapsed_sec,
        eta_sec=eta_sec,
        progress=progress,
        extra=status_extra,
    )

    if telemetry_metrics is None:
        return

    metric_payload = {
        "phase": "train",
        "loss": round(step_result.loss, 4),
        "learning_rate": learning_rate,
    }
    for key, value in step_result.metrics.items():
        metric_payload[key] = round(float(value), 4)
    for key, value in telemetry_metrics.items():
        metric_payload[key] = round(float(value), 4)

    events = [
        build_run_event(run_id, "metric", "training", step, metric_payload),
        build_run_event(
            run_id,
            "infra",
            "infra",
            step,
            {
                "device": "cpu",
                "step_time_ms": round(step_time_ms, 2),
                "samples_per_sec": round(batch_size / max(step_time_ms / 1000, 1e-6), 1),
                "memory_peak_mb": 0.0,
                "elapsed_sec": round(elapsed_sec, 2),
            },
        ),
    ]
    if layers:
        events.append(
            aggregate_layer_snapshot(
                run_store,
                run_id,
                step,
                layers,
                max_live_layer_samples=max_live_layer_samples,
            )
        )
    run_registry.publish(run_id, events)
