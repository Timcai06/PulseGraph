from __future__ import annotations

import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app.events.run_store import RunStore
from app.main import app
from app.reports.comparison import compare_runs
from app.schemas import RunEvent

client = TestClient(app)


def event(run_id: str, event_type: str, step: int, payload: dict, layer: str | None = None) -> RunEvent:
    return RunEvent(
        event_id=str(uuid.uuid4()),
        ts_ns=time.time_ns(),
        source="runtime_hook" if event_type == "layer_snapshot" else "training",
        type=event_type,
        run_id=run_id,
        step=step,
        epoch=1,
        layer=layer,
        payload=payload,
    )


def test_compare_runs_returns_contract_and_first_batch_divergence(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PULSEGRAPH_RUNS_DIR", str(tmp_path))
    store = RunStore()
    baseline = "healthy"
    candidate = "faulty"
    shared = {"task": "classification", "workload": "pet", "model": "resnet18", "seed": 42}
    store.save_config(baseline, {**shared, "mode": "healthy", "transform_fingerprint": "aaa"})
    store.save_config(candidate, {**shared, "mode": "bad-normalization", "transform_fingerprint": "bbb"})
    store.append(
        baseline,
        [
            event(baseline, "evidence", 1, {"kind": "batch", "sample_ids": ["pet-1"], "input_mean": -0.1, "input_std": 1.1}),
            event(baseline, "metric", 1, {"loss": 3.4, "accuracy": 0.1}),
        ],
    )
    store.append(
        candidate,
        [
            event(candidate, "evidence", 1, {"kind": "batch", "sample_ids": ["pet-1"], "input_mean": -1.4, "input_std": 5.3}),
            event(candidate, "metric", 1, {"loss": 3.5, "accuracy": 0.1}),
        ],
    )

    result = compare_runs(store, baseline, candidate)

    assert result is not None
    assert result.comparable is True
    assert {item.path for item in result.contract_differences} == {"mode", "transform_fingerprint"}
    assert result.first_observed_divergence is not None
    divergence = result.first_observed_divergence
    assert divergence.step == 1
    assert divergence.epoch == 1
    assert divergence.category == "batch"
    assert divergence.signal == "input_mean"
    assert divergence.layer is None
    assert divergence.baseline == -0.1
    assert divergence.candidate == -1.4
    assert divergence.absolute_delta == pytest.approx(1.3)
    assert divergence.relative_delta == pytest.approx(13.0)


def test_compare_runs_rejects_missing_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PULSEGRAPH_RUNS_DIR", str(tmp_path))
    assert compare_runs(RunStore(), "missing-a", "missing-b") is None


def test_compare_runs_api_returns_recorded_divergence() -> None:
    baseline = f"compare-base-{uuid.uuid4().hex[:8]}"
    candidate = f"compare-fault-{uuid.uuid4().hex[:8]}"
    config = {"task": "classification", "workload": "pet", "model": "resnet18", "seed": 42}
    for run_id, mean in ((baseline, -0.1), (candidate, -1.4)):
        response = client.post(
            f"/api/runs/{run_id}/events",
            json=[
                event(run_id, "config_registered", 0, {"config": config}).model_dump(mode="json"),
                event(run_id, "evidence", 1, {"kind": "batch", "sample_ids": ["pet-1"], "input_mean": mean}).model_dump(mode="json"),
                event(run_id, "run_complete", 1, {}).model_dump(mode="json"),
            ],
        )
        assert response.status_code == 200

    response = client.get(
        "/api/runs/compare",
        params={"baseline_run_id": baseline, "candidate_run_id": candidate},
    )

    assert response.status_code == 200
    assert response.json()["first_observed_divergence"]["signal"] == "input_mean"
    client.delete(f"/api/runs/{baseline}")
    client.delete(f"/api/runs/{candidate}")
