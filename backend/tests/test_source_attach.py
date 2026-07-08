import io
import uuid

import torch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

BLOCKS_SOURCE = """import torch
from torch import nn


class HiddenBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(28 * 28, 16)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.layer(x))
"""

ENTRY_SOURCE = """import torch
from torch import nn

from modules.blocks import HiddenBlock


class AttachedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.block = HiddenBlock()
        self.head = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(self.flatten(x)))
"""


def _build_attached_net() -> torch.nn.Module:
    namespace: dict = {}
    exec(BLOCKS_SOURCE, namespace)
    entry_namespace: dict = {
        "torch": torch,
        "nn": torch.nn,
        "HiddenBlock": namespace["HiddenBlock"],
    }
    exec(ENTRY_SOURCE.replace("from modules.blocks import HiddenBlock\n", ""), entry_namespace)
    return entry_namespace["AttachedNet"]()


def _source_upload_files():
    return [
        ("files", ("models/entry.py", ENTRY_SOURCE.encode(), "text/x-python")),
        ("files", ("modules/blocks.py", BLOCKS_SOURCE.encode(), "text/x-python")),
    ]


def test_candidates_analyzer_finds_module_classes_across_files() -> None:
    response = client.post("/api/inspect/source/candidates", files=_source_upload_files())

    assert response.status_code == 200
    payload = response.json()
    names = {(item["class_name"], item["file"]) for item in payload["candidates"]}
    assert ("AttachedNet", "models/entry.py") in names
    assert ("HiddenBlock", "modules/blocks.py") in names


def test_import_bare_pt_then_attach_multifile_source_then_replay() -> None:
    model = _build_attached_net()
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)

    imported = client.post("/api/runs/import", content=checkpoint.getvalue()).json()
    run_id = imported["run_id"]
    assert imported["created"] is True
    assert run_id.startswith("imported-")

    samples = io.BytesIO()
    torch.save({"images": torch.rand(4, 1, 28, 28), "labels": torch.tensor([1, 2, 3, 4])}, samples)
    client.post(f"/api/runs/{run_id}/samples", content=samples.getvalue())

    attach = client.post(
        f"/api/runs/{run_id}/source",
        files=_source_upload_files(),
        data={"entry_file": "models/entry.py", "entry_class": "AttachedNet"},
    ).json()
    assert sorted(attach["saved"]) == ["models/entry.py", "modules/blocks.py"]
    assert attach["validation"] == {"ok": True, "error": None, "missing_keys": [], "unexpected_keys": []}

    detail = client.get(f"/api/runs/{run_id}/detail").json()
    assert detail["source_origin"] == "user-attached"
    assert detail["entry_class"] == "AttachedNet"
    assert sorted(detail["source_files"]) == ["models/entry.py", "modules/blocks.py"]

    forward = client.get(f"/api/runs/{run_id}/forward?index=1").json()
    assert forward["label"] == 2
    assert len(forward["probabilities"]) == 10

    # re-importing the same weights should resolve to the existing run
    again = client.post("/api/runs/import", content=checkpoint.getvalue()).json()
    assert again == {"run_id": run_id, "fingerprint": imported["fingerprint"], "created": False}


def test_import_source_file_creates_replayable_streamable_run() -> None:
    response = client.post(
        "/api/runs/from-source",
        files=_source_upload_files(),
        data={"entry_file": "models/entry.py", "entry_class": "AttachedNet"},
    )

    assert response.status_code == 200
    created = response.json()
    run_id = created["run_id"]
    assert created["checkpoint"]["fingerprint"]
    assert created["graph"]["nodes"]

    detail = client.get(f"/api/runs/{run_id}/detail").json()
    assert detail["source_origin"] == "user-attached"
    assert detail["config"]["inference_only"] is True
    assert detail["config"]["run_kind"] == "source-import"
    assert detail["has_samples"] is True
    assert len(detail["checkpoints"]) == 1
    assert detail["event_count"] >= 4

    forward = client.get(f"/api/runs/{run_id}/forward?index=0").json()
    assert len(forward["probabilities"]) == 10
    assert forward["weights"] == "random"
    assert forward["sample_source"] == "probe"
    assert any(layer["layer_id"] == "block.layer" for layer in forward["layers"])

    stream = client.get(f"/api/runs/{run_id}/stream")
    assert "event: graph" in stream.text
    assert "event: run_complete" in stream.text
    assert "event: metric" not in stream.text
    assert "event: infra" not in stream.text


def test_train_source_file_creates_training_metrics_and_trained_forward() -> None:
    response = client.post(
        "/api/runs/train-source",
        files=_source_upload_files(),
        data={"entry_file": "models/entry.py", "entry_class": "AttachedNet", "steps": "4"},
    )

    assert response.status_code == 200
    trained = response.json()
    run_id = trained["run_id"]
    assert trained["run_kind"] == "source-training"
    assert trained["inference_only"] is False
    assert trained["status"] == "started"
    assert trained["checkpoint"] is None

    detail = client.get(f"/api/runs/{run_id}/detail").json()
    assert detail["config"]["run_kind"] == "source-training"
    assert detail["config"]["training_status"] == "completed"
    assert len(detail["metrics"]) >= 4

    forward = client.get(f"/api/runs/{run_id}/forward?index=0").json()
    assert len(forward["probabilities"]) == 10
    assert forward["weights"] == "trained"
    assert forward["sample_source"] == "probe"
    assert forward["label"] == 0
    assert len(forward["image_pixels"]) == 28 * 28
    assert max(forward["image_pixels"]) > 0.4

    stream = client.get(f"/api/runs/{run_id}/stream")
    assert "event: metric" in stream.text
    assert "event: infra" in stream.text
    assert "event: checkpoint" in stream.text
    assert "event: run_complete" in stream.text


def test_attach_reports_weight_mismatch() -> None:
    model = _build_attached_net()
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    run_id = client.post("/api/runs/import", content=checkpoint.getvalue()).json()["run_id"]

    wrong_entry = ENTRY_SOURCE.replace("nn.Linear(16, 10)", "nn.Linear(32, 10)").replace(
        "nn.Linear(28 * 28, 16)", "nn.Linear(28 * 28, 16)"
    )
    response = client.post(
        f"/api/runs/{run_id}/source",
        files=[
            ("files", ("models/entry.py", wrong_entry.encode(), "text/x-python")),
            ("files", ("modules/blocks.py", BLOCKS_SOURCE.encode(), "text/x-python")),
        ],
        data={"entry_file": "models/entry.py", "entry_class": "AttachedNet"},
    ).json()

    assert response["validation"]["ok"] is False
    assert "do not match" in response["validation"]["error"]


def test_attach_rejects_path_traversal_and_requires_entry() -> None:
    run_id = f"attach-{uuid.uuid4().hex[:8]}"
    response = client.post(
        f"/api/runs/{run_id}/source",
        files=[("files", ("../../evil.py", b"x = 1", "text/x-python"))],
        data={"entry_file": "../../evil.py", "entry_class": "X"},
    )

    assert response.status_code == 400
