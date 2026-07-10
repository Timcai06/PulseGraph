import io
import time
import uuid

import torch
from fastapi.testclient import TestClient

from app.main import app
from app.inspector.fingerprint import fingerprint_state_dict

client = TestClient(app)

TINY_SOURCE = """import torch
from torch import nn


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)
"""


def _event(run_id: str, event_type: str, payload: dict, step: int = 0) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "ts_ns": time.time_ns(),
        "source": "training",
        "type": event_type,
        "run_id": run_id,
        "step": step,
        "payload": payload,
    }


def _build_tiny_model() -> torch.nn.Module:
    namespace: dict = {}
    exec(TINY_SOURCE, namespace)
    # no fixed seed: each test's weights (and thus fingerprint) must be unique
    return namespace["TinyMLP"]()


def _register_full_run(run_id: str) -> bytes:
    """Register source/config/graph, upload samples and a checkpoint; returns checkpoint bytes."""
    client.post(
        f"/api/runs/{run_id}/events",
        json=[
            _event(
                run_id,
                "source_registered",
                {"classes": [{"name": "TinyMLP", "source_code": TINY_SOURCE}], "entry_class": "TinyMLP"},
            ),
            _event(run_id, "config_registered", {"config": {"lr": 0.001, "epochs": 1, "optimizer": "Adam"}}),
            _event(run_id, "graph_registered", {"graph": {"nodes": [], "edges": []}}),
        ],
    )
    model = _build_tiny_model()

    samples = io.BytesIO()
    images = torch.rand(6, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    model.eval()
    with torch.no_grad():
        predictions = model(images).argmax(dim=1).tolist()
    torch.save(
        {
            "images": images,
            "labels": labels,
            "predictions": [
                {"kind": "classification", "prediction": int(prediction)} for prediction in predictions
            ],
        },
        samples,
    )
    response = client.post(f"/api/runs/{run_id}/samples", content=samples.getvalue())
    assert response.status_code == 200

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    response = client.post(f"/api/runs/{run_id}/checkpoints?step=10&epoch=1", content=checkpoint.getvalue())
    assert response.status_code == 200
    assert response.json()["fingerprint"]
    return checkpoint.getvalue()


def test_registration_events_persist_and_appear_in_detail() -> None:
    run_id = f"prov-{uuid.uuid4().hex[:8]}"
    _register_full_run(run_id)

    detail = client.get(f"/api/runs/{run_id}/detail").json()

    assert detail["entry_class"] == "TinyMLP"
    assert "class TinyMLP" in detail["source"]
    assert detail["config"]["optimizer"] == "Adam"
    assert detail["has_samples"] is True
    assert len(detail["checkpoints"]) == 1
    assert detail["checkpoints"][0]["step"] == 10
    assert client.get(f"/api/runs/{run_id}/source").text.startswith("import torch")


def test_uploaded_pt_matches_recorded_run_by_fingerprint() -> None:
    run_id = f"prov-{uuid.uuid4().hex[:8]}"
    checkpoint_bytes = _register_full_run(run_id)

    response = client.post(
        "/api/inspect/upload",
        files={"file": ("mystery.pt", checkpoint_bytes, "application/octet-stream")},
    )

    payload = response.json()
    assert payload["matched_run_id"] == run_id
    assert payload["weights_fingerprint"]
    assert "full training provenance" in payload["warnings"][0]


def test_fingerprint_supports_scalar_integer_buffers() -> None:
    fingerprint = fingerprint_state_dict(
        {
            "weight": torch.ones(2, 2),
            "num_batches_tracked": torch.tensor(3, dtype=torch.int64),
        }
    )

    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 64


def test_forward_replay_rebuilds_model_from_source_and_checkpoint() -> None:
    run_id = f"prov-{uuid.uuid4().hex[:8]}"
    _register_full_run(run_id)

    response = client.get(f"/api/runs/{run_id}/forward?checkpoint_step=10&index=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == 2  # from probe samples
    assert len(payload["probabilities"]) == 10
    assert abs(sum(payload["probabilities"]) - 1.0) < 1e-4
    assert any(layer["layer_id"] == "net.1" for layer in payload["layers"])


def test_forward_replay_requires_recorded_source() -> None:
    run_id = f"prov-{uuid.uuid4().hex[:8]}"
    client.post(f"/api/runs/{run_id}/events", json=_event(run_id, "metric", {"loss": 1.0}, step=1))

    response = client.get(f"/api/runs/{run_id}/forward")

    assert response.status_code == 400
    assert "no recorded model source" in response.json()["detail"]


def test_report_contains_insights_and_checkpoint_evaluations() -> None:
    run_id = f"prov-{uuid.uuid4().hex[:8]}"
    _register_full_run(run_id)
    metric_events = [
        _event(run_id, "metric", {"loss": 2.0 - i * 0.1, "accuracy": 0.1 + i * 0.05, "phase": "train"}, step=i * 10)
        for i in range(1, 11)
    ]
    metric_events.append(_event(run_id, "metric", {"loss": 1.2, "accuracy": 0.4, "phase": "eval"}, step=100))
    layer_events = [
        _event(run_id, "layer_snapshot", {"activation_sparsity": 0.95, "gradient_norm": 0.5, "weight_std": 0.02}, step=i * 10)
        for i in range(1, 5)
    ]
    for event in layer_events:
        event["layer"] = "net.1"
        event["source"] = "runtime_hook"
    client.post(f"/api/runs/{run_id}/events", json=metric_events + layer_events)

    report = client.get(f"/api/runs/{run_id}/report").json()

    assert report["best_accuracy"] is not None
    assert report["overfit_gap"] is not None and report["overfit_gap"] > 0.05
    assert report["checkpoint_evaluations"] and report["checkpoint_evaluations"][0]["sample_count"] == 6
    assert report["error_analysis"] is not None
    titles = [insight["title"] for insight in report["insights"]]
    assert any("Overfitting" in title for title in titles)
    assert any("mostly inactive" in title for title in titles)
