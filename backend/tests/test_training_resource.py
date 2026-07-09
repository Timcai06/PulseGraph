import textwrap

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.resources.contract import load_training_resource
from app.runtime import mnist_data


client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_mnist_caches():
    mnist_data.load_train_samples.cache_clear()
    mnist_data.load_test_samples.cache_clear()
    yield
    mnist_data.load_train_samples.cache_clear()
    mnist_data.load_test_samples.cache_clear()


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

RGB_RESOURCE_SOURCE = textwrap.dedent(
    """
    import torch
    from torch import nn


    def metadata():
        return {
            "name": "rgb_resource",
            "classes": 3,
            "class_names": ["red", "green", "blue"],
            "input_shape": [3, 2, 2],
            "batch_size": 4,
        }


    def build_model():
        return nn.Sequential(nn.Flatten(), nn.Linear(12, 3))


    def train_batch(step, batch_size):
        images = torch.zeros(batch_size, 3, 2, 2)
        labels = torch.arange(batch_size) % 3
        for row, label in enumerate(labels):
            images[row, label, :, :] = 1.0
        return images, labels


    def inference_sample(index):
        image = torch.zeros(3, 2, 2)
        label = index % 3
        image[label, :, :] = 1.0
        return image, label
    """
)

RGB_REPORT_RESOURCE_SOURCE = textwrap.dedent(
    """
    import torch
    from torch import nn


    class FixedRgbNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor([10.0, 0.0, 0.0]))

        def forward(self, images):
            return self.bias.unsqueeze(0).repeat(images.shape[0], 1)


    def metadata():
        return {
            "name": "rgb_report_resource",
            "classes": 3,
            "class_names": ["red", "green", "blue"],
            "input_shape": [3, 2, 2],
            "batch_size": 4,
        }


    def build_model():
        return FixedRgbNet()


    def train_batch(step, batch_size):
        images = torch.zeros(batch_size, 3, 2, 2)
        labels = torch.tensor([(index % 2) + 1 for index in range(batch_size)])
        for row, label in enumerate(labels):
            images[row, label, :, :] = 1.0
        return images, labels


    def inference_sample(index):
        label = (index % 2) + 1
        image = torch.zeros(3, 2, 2)
        image[label, :, :] = 1.0
        return image, label
    """
)

BAD_CLASS_NAMES_SOURCE = RGB_RESOURCE_SOURCE.replace(
    '"class_names": ["red", "green", "blue"]',
    '"class_names": ["red", "green"]',
)

BAD_IMAGE_SHAPE_SOURCE = RGB_RESOURCE_SOURCE.replace(
    "image = torch.zeros(3, 2, 2)",
    "image = torch.zeros(2, 2, 2)",
)

ORDINARY_MODULE_SOURCE = textwrap.dedent(
    """
    import torch
    from torch import nn


    class DigitMLP(nn.Module):
        def __init__(self):
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
)


def test_training_resource_contract_loads_model_batches_and_samples(tmp_path) -> None:
    resource_path = tmp_path / "resource.py"
    resource_path.write_text(RESOURCE_SOURCE, encoding="utf-8")

    resource = load_training_resource(resource_path)

    assert resource.metadata["name"] == "tiny_resource"
    assert resource.input_shape == [1, 1, 4]
    assert resource.classes == 3
    assert resource.build_model().__class__.__name__ == "Linear"
    images, labels = resource.train_batch(step=2, batch_size=5)
    assert list(images.shape) == [5, 4]
    assert labels.tolist() == [0, 1, 2, 0, 1]
    sample, label = resource.inference_sample(index=2)
    assert list(sample.shape) == [1, 4]
    assert label == 2


def test_training_resource_accepts_named_rgb_image_resource(tmp_path) -> None:
    resource_path = tmp_path / "resource.py"
    resource_path.write_text(RGB_RESOURCE_SOURCE, encoding="utf-8")

    resource = load_training_resource(resource_path)

    assert resource.input_shape == [3, 2, 2]
    assert resource.classes == 3
    assert resource.metadata["class_names"] == ["red", "green", "blue"]
    sample, label = resource.inference_sample(index=1)
    assert list(sample.shape) == [3, 2, 2]
    assert label == 1


def test_training_resource_rejects_class_name_count_mismatch(tmp_path) -> None:
    resource_path = tmp_path / "resource.py"
    resource_path.write_text(BAD_CLASS_NAMES_SOURCE, encoding="utf-8")

    with pytest.raises(ValueError, match="class_names"):
        load_training_resource(resource_path)


def test_training_resource_rejects_non_image_sample_shape(tmp_path) -> None:
    resource_path = tmp_path / "resource.py"
    resource_path.write_text(BAD_IMAGE_SHAPE_SOURCE, encoding="utf-8")

    with pytest.raises(ValueError, match="C,H,W"):
        load_training_resource(resource_path)


def test_plain_nn_module_source_is_adapted_as_training_resource(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PULSEGRAPH_MNIST_DIR", str(tmp_path / "missing-mnist"))
    mnist_data.load_train_samples.cache_clear()
    mnist_data.load_test_samples.cache_clear()
    resource_path = tmp_path / "mlp.py"
    resource_path.write_text(ORDINARY_MODULE_SOURCE, encoding="utf-8")

    resource = load_training_resource(resource_path)

    assert resource.name == "DigitMLP"
    assert resource.metadata["source_mode"] == "nn-module-adapter"
    assert resource.metadata["data_source"] == "synthetic"
    assert resource.input_shape == [1, 28, 28]
    assert resource.classes == 10
    images, labels = resource.train_batch(step=1, batch_size=4)
    assert list(images.shape) == [4, 1, 28, 28]
    assert labels.tolist() == [1, 2, 3, 4]
    sample, label = resource.inference_sample(index=5)
    assert list(sample.shape) == [1, 1, 28, 28]
    assert label == 5


def test_plain_nn_module_uses_real_mnist_when_available(tmp_path) -> None:
    if mnist_data.load_train_samples() is None or mnist_data.load_test_samples() is None:
        return
    resource_path = tmp_path / "mlp.py"
    resource_path.write_text(ORDINARY_MODULE_SOURCE, encoding="utf-8")

    resource = load_training_resource(resource_path)

    assert resource.metadata["data_source"] == "mnist"
    images, labels = resource.train_batch(step=1, batch_size=4)
    assert list(images.shape) == [4, 1, 28, 28]
    assert labels.min().item() >= 0
    assert labels.max().item() <= 9
    sample, label = resource.inference_sample(index=0)
    assert list(sample.shape) == [1, 1, 28, 28]
    assert 0 <= label <= 9


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


def test_train_resource_endpoint_transmits_class_names_and_image_shape() -> None:
    preview = client.post(
        "/api/inspect/resource/preview",
        files=[("files", ("resource.py", RGB_RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py"},
    )

    assert preview.status_code == 200
    assert preview.json()["resource"]["class_names"] == ["red", "green", "blue"]
    assert preview.json()["resource"]["input_shape"] == [3, 2, 2]
    samples = preview.json()["samples"]
    assert samples[0]["label_name"] == "red"
    assert samples[0]["image_shape"] == [3, 2, 2]
    assert len(samples[0]["image_pixels"]) == 12

    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("resource.py", RGB_RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py", "steps": "2"},
    )

    assert response.status_code == 200
    payload = response.json()
    run_id = payload["run_id"]
    assert payload["resource"]["class_names"] == ["red", "green", "blue"]

    detail = client.get(f"/api/runs/{run_id}/detail").json()
    assert detail["config"]["class_names"] == ["red", "green", "blue"]

    forward = client.get(f"/api/runs/{run_id}/forward?index=2").json()
    assert forward["class_names"] == ["red", "green", "blue"]
    assert forward["image_shape"] == [3, 2, 2]
    assert len(forward["image_pixels"]) == 12


def test_resource_report_contains_named_rgb_misclassified_samples() -> None:
    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("resource.py", RGB_REPORT_RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py", "steps": "1"},
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    report = client.get(f"/api/runs/{run_id}/report").json()

    assert report["error_analysis"]["class_names"] == ["red", "green", "blue"]
    assert report["error_analysis"]["labels"] == [0, 1, 2]
    sample = report["error_analysis"]["misclassified"][0]
    assert sample["label_name"] in {"green", "blue"}
    assert sample["prediction_name"] == "red"
    assert sample["image_shape"] == [3, 2, 2]
    assert len(sample["pixels"]) == 12


def test_train_resource_endpoint_respects_telemetry_stride() -> None:
    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("resource.py", RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py", "steps": "6", "telemetry_stride": "3"},
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    detail = client.get(f"/api/runs/{run_id}/detail").json()

    assert [metric["step"] for metric in detail["metrics"]] == [3, 6]
    assert detail["config"]["steps"] == 6
    assert detail["config"]["telemetry_stride"] == 3


def test_train_resource_endpoint_accepts_plain_nn_module_source() -> None:
    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("mlp.py", ORDINARY_MODULE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "mlp.py", "steps": "2"},
    )

    assert response.status_code == 200
    payload = response.json()
    run_id = payload["run_id"]
    assert payload["run_kind"] == "resource-training"
    assert payload["resource"]["name"] == "DigitMLP"

    detail = client.get(f"/api/runs/{run_id}/detail").json()
    assert detail["config"]["resource_name"] == "DigitMLP"
    assert len(detail["metrics"]) >= 2

    forward = client.get(f"/api/runs/{run_id}/forward?index=3").json()
    assert 0 <= forward["label"] <= 9
    assert len(forward["probabilities"]) == 10


def test_plain_nn_module_forward_reports_mnist_source_when_real_samples_exist() -> None:
    if mnist_data.load_train_samples() is None or mnist_data.load_test_samples() is None:
        return
    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("mlp.py", ORDINARY_MODULE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "mlp.py", "steps": "2"},
    )
    run_id = response.json()["run_id"]

    forward = client.get(f"/api/runs/{run_id}/forward?index=3").json()

    assert forward["sample_source"] == "mnist"
    assert len(forward["probabilities"]) == 10


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
