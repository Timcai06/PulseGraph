import textwrap

from fastapi.testclient import TestClient

from app.main import app
from app.resources.contract import load_training_resource


client = TestClient(app)


RESOURCE_SOURCE = textwrap.dedent(
    """
    import torch
    from torch import nn


    def metadata():
        return {"name": "tiny_resource", "classes": 3, "input_shape": [1, 4]}


    def build_model():
        return nn.Linear(4, 3)


    def train_batch(step, batch_size):
        images = torch.ones(batch_size, 4) * step
        labels = torch.arange(batch_size) % 3
        return images, labels


    def inference_sample(index):
        return torch.ones(1, 4) * index, index % 3
    """
)


def test_training_resource_contract_loads_model_batches_and_samples(tmp_path) -> None:
    resource_path = tmp_path / "resource.py"
    resource_path.write_text(RESOURCE_SOURCE, encoding="utf-8")

    resource = load_training_resource(resource_path)

    assert resource.metadata["name"] == "tiny_resource"
    assert resource.input_shape == [4]
    assert resource.classes == 3
    assert resource.build_model().__class__.__name__ == "Linear"
    images, labels = resource.train_batch(step=2, batch_size=5)
    assert list(images.shape) == [5, 4]
    assert labels.tolist() == [0, 1, 2, 0, 1]
    sample, label = resource.inference_sample(index=2)
    assert list(sample.shape) == [1, 4]
    assert label == 2


def test_train_resource_endpoint_creates_run_and_forward_replay() -> None:
    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("resource.py", RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py", "steps": "2"},
    )

    assert response.status_code == 200
    payload = response.json()
    run_id = payload["run_id"]
    assert payload["run_kind"] == "resource-training"
    assert payload["resource"]["name"] == "tiny_resource"

    detail = client.get(f"/api/runs/{run_id}/detail").json()
    assert detail["config"]["run_kind"] == "resource-training"
    assert detail["config"]["resource_name"] == "tiny_resource"
    assert len(detail["metrics"]) >= 2

    forward = client.get(f"/api/runs/{run_id}/forward?index=1").json()
    assert forward["label"] == 1
    assert len(forward["probabilities"]) == 3


def test_delete_run_removes_history_detail_and_files() -> None:
    created = client.post(
        "/api/runs/train-resource",
        files=[("files", ("resource.py", RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py", "steps": "1"},
    ).json()
    run_id = created["run_id"]

    response = client.delete(f"/api/runs/{run_id}")

    assert response.status_code == 200
    assert response.json() == {"deleted": True, "run_id": run_id}
    assert client.get(f"/api/runs/{run_id}/detail").status_code == 404
    assert all(run["run_id"] != run_id for run in client.get("/api/runs").json())
