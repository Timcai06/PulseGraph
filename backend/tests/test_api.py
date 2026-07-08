import torch
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.schemas import RunEvent


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_demo_forward_route() -> None:
    response = client.get("/api/demo/forward?index=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == 2
    assert len(payload["probabilities"]) == 10


def test_upload_inspection(tmp_path) -> None:
    model_path = tmp_path / "tiny.pt"
    torch.save({"linear.weight": torch.randn(3, 2), "linear.bias": torch.randn(3)}, model_path)

    with model_path.open("rb") as handle:
        response = client.post("/api/inspect/upload", files={"file": ("tiny.pt", handle, "application/octet-stream")})

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "tiny.pt"
    assert payload["artifact_id"].startswith("pulsegraph:sha256:")
    assert len(payload["artifact_sha256"]) == 64
    assert payload["mode"] == "state_dict"
    assert payload["graph"]["nodes"][1]["kind"] == "Linear"


def test_upload_rejects_oversized_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 4)

    response = client.post(
        "/api/inspect/upload",
        files={"file": ("too-large.pt", b"12345", "application/octet-stream")},
    )

    assert response.status_code == 413


def test_stream_demo_run_endpoint(monkeypatch) -> None:
    async def fake_events():
        yield RunEvent(
            event_id="event-1",
            ts_ns=1,
            source="training",
            type="metric",
            run_id="test-run",
            step=1,
            epoch=1,
            payload={"loss": 1.0, "accuracy": 0.1},
        )

    monkeypatch.setattr(main_module, "demo_training_events", fake_events)

    with client.stream("GET", "/api/runs/demo/stream") as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: metric" in body
    assert '"schema_version": "pulsegraph.event.v1"' in body
