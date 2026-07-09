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


TINY_RESOURCE = """
import torch
from torch import nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(self.flatten(x))
"""

MISSING_IMPORT_RESOURCE = "from datasets.mnist import load_mnist\n"


def test_resource_preview_returns_graph_without_creating_a_run() -> None:
    response = client.post(
        "/api/inspect/resource/preview",
        files=[("files", ("tiny.py", TINY_RESOURCE.encode(), "text/x-python"))],
        data={"entry_file": "tiny.py"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["resource"]["name"] == "TinyNet"
    assert payload["resource"]["classes"] == 10
    assert any(node["kind"] == "Linear" for node in payload["graph"]["nodes"])


def test_missing_import_is_a_400_with_guidance_not_a_500() -> None:
    for endpoint in ("/api/inspect/resource/preview", "/api/runs/train-resource"):
        response = client.post(
            endpoint,
            files=[("files", ("broken.py", MISSING_IMPORT_RESOURCE.encode(), "text/x-python"))],
            data={"entry_file": "broken.py"},
        )

        assert response.status_code == 400, endpoint
        detail = response.json()["detail"]
        assert "datasets" in detail
        assert ".zip" in detail


def test_entry_resolution_for_zip_uploads_falls_back_to_resource_py() -> None:
    # the browser sends the archive name as entry_file; the backend only sees
    # the expanded .py files, so it must fall back to the conventional root
    response = client.post(
        "/api/inspect/resource/preview",
        files=[
            ("files", ("resource.py", TINY_RESOURCE.encode(), "text/x-python")),
            ("files", ("models/extra.py", b"", "text/x-python")),
        ],
        data={"entry_file": "resource.zip"},
    )

    assert response.status_code == 200
    assert response.json()["entry_file"] == "resource.py"


def test_resource_preview_resolves_packaged_imports() -> None:
    entry = b"from nets.core import TinyNet\n\ndef build_model():\n    return TinyNet()\n"
    response = client.post(
        "/api/inspect/resource/preview",
        files=[
            ("files", ("entry.py", entry, "text/x-python")),
            ("files", ("nets/__init__.py", b"", "text/x-python")),
            ("files", ("nets/core.py", TINY_RESOURCE.encode(), "text/x-python")),
        ],
        data={"entry_file": "entry.py"},
    )

    assert response.status_code == 200
    assert response.json()["resource"]["classes"] == 10


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
