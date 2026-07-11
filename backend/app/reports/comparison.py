from __future__ import annotations

from typing import Any

from app.events.run_store import RunStore
from app.schemas import RunComparison, RunEvent, RunValueDifference, SignalDivergence

NUMERIC_SIGNALS = {
    "input_mean",
    "input_std",
    "loss",
    "accuracy",
    "learning_rate",
    "activation_mean",
    "activation_sparsity",
    "gradient_norm",
    "weight_std",
}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            flattened.update(_flatten(child, path))
        else:
            flattened[path] = child
    return flattened


def _contract_differences(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[RunValueDifference]:
    left = _flatten(baseline)
    right = _flatten(candidate)
    differences = [
        RunValueDifference(path=path, baseline=left.get(path), candidate=right.get(path))
        for path in sorted(set(left) | set(right))
        if left.get(path) != right.get(path)
    ]
    return differences[:64]


def _event_key(event: RunEvent) -> tuple[int, int | None, str, str | None]:
    kind = str(event.payload.get("kind") or "") if event.type == "evidence" else event.type
    return event.step, event.epoch, kind, event.layer


def _paired_events(baseline: list[RunEvent], candidate: list[RunEvent], event_type: str):
    left = {_event_key(event): event for event in baseline if event.type == event_type}
    right = {_event_key(event): event for event in candidate if event.type == event_type}
    for key in sorted(set(left) & set(right), key=lambda item: (item[0], item[2], item[3] or "")):
        yield left[key], right[key]


def _difference(
    baseline: RunEvent,
    candidate: RunEvent,
    category: str,
    signal: str,
    priority: int,
) -> tuple[int, int, SignalDivergence] | None:
    left = baseline.payload.get(signal)
    right = candidate.payload.get(signal)
    if left == right:
        return None
    absolute_delta = None
    relative_delta = None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        absolute_delta = abs(float(right) - float(left))
        if absolute_delta <= 1e-8:
            return None
        relative_delta = absolute_delta / max(abs(float(left)), 1e-12)
    return (
        baseline.step,
        priority,
        SignalDivergence(
            step=baseline.step,
            epoch=baseline.epoch,
            category=category,
            signal=signal,
            layer=baseline.layer,
            baseline=left,
            candidate=right,
            absolute_delta=absolute_delta,
            relative_delta=relative_delta,
        ),
    )


def _first_observed_divergence(baseline: list[RunEvent], candidate: list[RunEvent]) -> SignalDivergence | None:
    found: list[tuple[int, int, SignalDivergence]] = []
    for left, right in _paired_events(baseline, candidate, "evidence"):
        if left.payload.get("kind") != "batch":
            continue
        for signal in ("sample_ids", "input_mean", "input_std"):
            difference = _difference(left, right, "batch", signal, 0)
            if difference:
                found.append(difference)
    for left, right in _paired_events(baseline, candidate, "layer_snapshot"):
        for signal in sorted(NUMERIC_SIGNALS & set(left.payload) & set(right.payload)):
            difference = _difference(left, right, "layer", signal, 1)
            if difference:
                found.append(difference)
    for left, right in _paired_events(baseline, candidate, "metric"):
        for signal in sorted(NUMERIC_SIGNALS & set(left.payload) & set(right.payload)):
            difference = _difference(left, right, "metric", signal, 2)
            if difference:
                found.append(difference)
    return min(found, key=lambda item: (item[0], item[1], item[2].signal))[2] if found else None


def compare_runs(store: RunStore, baseline_run_id: str, candidate_run_id: str) -> RunComparison | None:
    baseline_events = store.load_events(baseline_run_id)
    candidate_events = store.load_events(candidate_run_id)
    if not baseline_events or not candidate_events:
        return None
    baseline_config = store.load_config(baseline_run_id) or {}
    candidate_config = store.load_config(candidate_run_id) or {}
    comparable = all(
        baseline_config.get(key) == candidate_config.get(key)
        for key in ("task", "workload", "model", "seed")
    )
    return RunComparison(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        comparable=comparable,
        contract_differences=_contract_differences(baseline_config, candidate_config),
        first_observed_divergence=_first_observed_divergence(baseline_events, candidate_events),
    )
