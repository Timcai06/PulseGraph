import io
import threading
import textwrap
import time
import uuid
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.main import app
from app.inspector.graph_builder import build_bounded_graph_from_tensor_specs
from app.resources.contract import load_training_resource
from app.runtime import mnist_data
from app.runtime.resource_training import run_resource_training_job
from app.schemas import ModelGraph


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

FAILING_RESOURCE_SOURCE = textwrap.dedent(
    """
    import torch
    from torch import nn


    def metadata():
        return {"name": "failing_resource", "classes": 3, "input_shape": [1, 4]}


    def build_model():
        return nn.Linear(4, 3)


    def train_batch(step, batch_size):
        if step == 2:
            raise RuntimeError("resource exploded at step 2")
        images = torch.ones(batch_size, 4) * step
        labels = torch.arange(batch_size) % 3
        return images, labels


    def inference_sample(index):
        return torch.ones(1, 4) * index, index % 3
    """
)

SLOW_RESOURCE_SOURCE = textwrap.dedent(
    """
    import time

    import torch
    from torch import nn


    def metadata():
        return {"name": "slow_resource", "classes": 3, "input_shape": [1, 4]}


    def build_model():
        return nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))


    def train_batch(step, batch_size):
        time.sleep(0.03)
        images = torch.ones(batch_size, 4) * step
        labels = torch.arange(batch_size) % 3
        return images, labels


    def inference_sample(index):
        return torch.ones(1, 4) * index, index % 3
    """
)

MANY_LAYER_RESOURCE_SOURCE = textwrap.dedent(
    """
    import torch
    from torch import nn


    class DeepTinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            for _ in range(70):
                blocks.append(nn.Linear(4, 4))
                blocks.append(nn.ReLU())
            blocks.append(nn.Linear(4, 3))
            self.net = nn.Sequential(*blocks)

        def forward(self, x):
            return self.net(x)


    def metadata():
        return {"name": "many_layer_resource", "classes": 3, "input_shape": [1, 4]}


    def build_model():
        return DeepTinyNet()


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
            "task": "classification",
            "dataset": {
                "kind": "image_classification",
                "name": "tiny_rgb",
                "source": "synthetic-rgb",
                "splits": ["train", "preview"],
            },
            "output_schema": {
                "kind": "classification",
                "renderer": "probability_chart",
            },
            "metric_schema": {
                "primary": "accuracy",
                "monitors": ["loss", "accuracy"],
            },
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

PACKAGE_RESOURCE_SOURCE = textwrap.dedent(
    """
    import torch

    from models.cifar import TinyPackagedNet


    def metadata():
        return {
            "name": "packaged_rgb_resource",
            "classes": 3,
            "class_names": ["red", "green", "blue"],
            "input_shape": [3, 2, 2],
            "batch_size": 4,
        }


    def build_model():
        return TinyPackagedNet()


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

PACKAGE_MODEL_SOURCE = textwrap.dedent(
    """
    from torch import nn


    class TinyPackagedNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))

        def forward(self, images):
            return self.net(images)
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

ASSET_RESOURCE_SOURCE = textwrap.dedent(
    """
    import json
    from pathlib import Path

    import torch
    from torch import nn
    from torchvision.io import read_image


    ROOT = Path(__file__).resolve().parent


    def metadata():
        return {
            "name": "asset_resource",
            "classes": 2,
            "class_names": ["background", "positive"],
            "input_shape": [3, 2, 2],
            "batch_size": 2,
        }


    def build_model():
        return nn.Sequential(nn.Flatten(), nn.Linear(12, 2))


    def _sample():
        image = read_image(str(ROOT / "fixtures" / "pixel.png")).float() / 255.0
        label = int(json.loads((ROOT / "fixtures" / "meta.json").read_text())["label"])
        return image, label


    def train_batch(step, batch_size):
        image, label = _sample()
        images = torch.stack([image for _ in range(batch_size)], dim=0)
        labels = torch.tensor([label for _ in range(batch_size)], dtype=torch.long)
        return images, labels


    def inference_sample(index):
        return _sample()
    """
)

GRAPH_SPEC_RESOURCE_SOURCE = textwrap.dedent(
    """
    import torch
    from torch import nn


    def metadata():
        return {"name": "graph_spec_resource", "classes": 2, "input_shape": [1, 1, 4]}


    def build_model():
        return nn.Sequential(nn.Flatten(), nn.Linear(4, 2))


    def train_batch(step, batch_size):
        images = torch.ones(batch_size, 1, 4)
        labels = torch.arange(batch_size) % 2
        return images, labels


    def inference_sample(index):
        return torch.ones(1, 4) * index, index % 2


    def graph_spec():
        return {
            "nodes": [
                {"id": "input", "label": "Input", "kind": "Input", "confidence": "trusted", "metadata": {}},
                {"id": "backbone", "label": "Backbone", "kind": "DetectorBackbone", "confidence": "trusted", "metadata": {"stage": "summary"}},
                {"id": "head", "label": "Head", "kind": "ClassifierHead", "confidence": "trusted", "metadata": {"stage": "summary"}},
            ],
            "edges": [
                {"id": "input->backbone", "source": "input", "target": "backbone"},
                {"id": "backbone->head", "source": "backbone", "target": "head"},
            ],
        }
    """
)

MANY_LAYER_RESOURCE_SOURCE = textwrap.dedent(
    """
    import torch
    from torch import nn


    class ManyLayerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(32)])
            self.head = nn.Linear(4, 2)

        def forward(self, images):
            output = self.flatten(images)
            for layer in self.layers:
                output = torch.relu(layer(output))
            return self.head(output)


    def metadata():
        return {"name": "many_layer_resource", "classes": 2, "input_shape": [1, 1, 4]}


    def build_model():
        return ManyLayerNet()


    def train_batch(step, batch_size):
        images = torch.ones(batch_size, 1, 4)
        labels = torch.arange(batch_size) % 2
        return images, labels


    def inference_sample(index):
        return torch.ones(1, 4) * index, index % 2
    """
)


def _zip_bytes(files: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, content in files:
            archive.writestr(path, content)
    return buffer.getvalue()


def _tiny_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (240, 48, 48)).save(buffer, format="PNG")
    return buffer.getvalue()


def _wait_for_condition(predicate, timeout_sec: float = 3.0) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for background condition.")


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
    assert detail["config"]["task"] == "classification"
    assert len(detail["metrics"]) >= 2

    forward = client.get(f"/api/runs/{run_id}/forward?index=1").json()
    assert forward["task"] == "classification"
    assert forward["output"]["kind"] == "classification"
    assert forward["output"]["prediction"] == forward["prediction"]
    assert forward["output"]["probabilities"] == forward["probabilities"]
    assert forward["label"] == 1
    assert len(forward["probabilities"]) == 3


def test_train_resource_endpoint_transmits_class_names_and_image_shape() -> None:
    preview = client.post(
        "/api/inspect/resource/preview",
        files=[("files", ("resource.py", RGB_RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py"},
    )

    assert preview.status_code == 200
    assert preview.json()["resource"]["task"] == "classification"
    assert preview.json()["resource"]["dataset_spec"]["name"] == "tiny_rgb"
    assert preview.json()["resource"]["dataset_spec"]["splits"] == ["train", "preview"]
    assert preview.json()["resource"]["output_schema"]["renderer"] == "probability_chart"
    assert preview.json()["resource"]["metric_schema"]["primary"] == "accuracy"
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
    assert payload["resource"]["task"] == "classification"
    assert payload["resource"]["dataset_spec"]["source"] == "synthetic-rgb"
    assert payload["resource"]["output_schema"]["kind"] == "classification"
    assert payload["resource"]["metric_schema"]["monitors"] == ["loss", "accuracy"]
    assert payload["resource"]["class_names"] == ["red", "green", "blue"]

    detail = client.get(f"/api/runs/{run_id}/detail").json()
    assert detail["config"]["task"] == "classification"
    assert detail["config"]["dataset_spec"]["name"] == "tiny_rgb"
    assert detail["config"]["output_schema"]["renderer"] == "probability_chart"
    assert detail["config"]["metric_schema"]["primary"] == "accuracy"
    assert detail["config"]["class_names"] == ["red", "green", "blue"]

    forward = client.get(f"/api/runs/{run_id}/forward?index=2").json()
    assert forward["task"] == "classification"
    assert forward["output"]["kind"] == "classification"
    assert forward["class_names"] == ["red", "green", "blue"]
    assert forward["image_shape"] == [3, 2, 2]
    assert len(forward["image_pixels"]) == 12


def test_train_resource_endpoint_accepts_folder_upload_with_package_imports() -> None:
    response = client.post(
        "/api/runs/train-resource",
        files=[
            ("files", ("cifar_rgb/resource.py", PACKAGE_RESOURCE_SOURCE.encode(), "text/x-python")),
            ("files", ("cifar_rgb/models/__init__.py", b"from models.cifar import TinyPackagedNet\n", "text/x-python")),
            ("files", ("cifar_rgb/models/cifar.py", PACKAGE_MODEL_SOURCE.encode(), "text/x-python")),
        ],
        data={"entry_file": "resource.py", "steps": "1"},
    )

    assert response.status_code == 200
    payload = response.json()
    run_id = payload["run_id"]
    assert payload["entry_file"] == "cifar_rgb/resource.py"
    assert payload["resource"]["name"] == "packaged_rgb_resource"

    forward = client.get(f"/api/runs/{run_id}/forward?index=2").json()
    assert forward["label"] == 2
    assert forward["class_names"] == ["red", "green", "blue"]


def test_train_resource_endpoint_accepts_packaged_binary_assets_and_hides_them_from_source_listing() -> None:
    archive = _zip_bytes(
        [
            ("resource.py", ASSET_RESOURCE_SOURCE.encode()),
            ("fixtures/meta.json", b'{"label": 1}'),
            ("fixtures/pixel.png", _tiny_png_bytes()),
            ("README.md", b"ignored for runtime"),
        ]
    )

    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("asset_resource.zip", archive, "application/zip"))],
        data={"entry_file": "asset_resource.zip", "steps": "1"},
    )

    assert response.status_code == 200
    payload = response.json()
    run_id = payload["run_id"]
    assert payload["entry_file"] == "resource.py"
    assert sorted(payload["saved"]) == ["fixtures/meta.json", "fixtures/pixel.png", "resource.py"]

    detail = client.get(f"/api/runs/{run_id}/detail").json()
    assert detail["source_files"] == ["resource.py"]

    forward = client.get(f"/api/runs/{run_id}/forward?index=0").json()
    assert forward["label"] == 1
    assert forward["image_shape"] == [3, 2, 2]


def test_resource_preview_counts_packaged_assets() -> None:
    archive = _zip_bytes(
        [
            ("resource.py", ASSET_RESOURCE_SOURCE.encode()),
            ("fixtures/meta.json", b'{"label": 1}'),
            ("fixtures/pixel.png", _tiny_png_bytes()),
        ]
    )

    response = client.post(
        "/api/inspect/resource/preview",
        files=[("files", ("asset_resource.zip", archive, "application/zip"))],
        data={"entry_file": "asset_resource.zip"},
    )

    assert response.status_code == 200
    assert sorted(response.json()["files"]) == ["fixtures/meta.json", "fixtures/pixel.png", "resource.py"]


def test_resource_preview_uses_graph_spec_when_fx_trace_fails(monkeypatch) -> None:
    def explode(*_args, **_kwargs):
        raise RuntimeError("fx detector tracing failed")

    monkeypatch.setattr(main_module, "trace_model_graph", explode)

    response = client.post(
        "/api/inspect/resource/preview",
        files=[("files", ("resource.py", GRAPH_SPEC_RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["graph_diagnostics"]["strategy"] == "resource_graph_spec"
    assert payload["graph_diagnostics"]["readability"] == "high"
    assert "fx trace failed" in payload["graph_diagnostics"]["note"]
    assert [node["id"] for node in payload["graph"]["nodes"]] == ["input", "backbone", "head"]
    assert all(node["confidence"] == "inferred" for node in payload["graph"]["nodes"])
    assert all(node["metadata"]["self_reported"] is True for node in payload["graph"]["nodes"])


def test_resource_preview_groups_oversized_fx_graph_when_no_graph_spec(monkeypatch) -> None:
    huge_graph = ModelGraph.model_validate(
        {
            "nodes": [
                {
                    "id": f"node_{index}",
                    "label": f"Node {index}",
                    "kind": "Linear",
                    "param_count": 4,
                    "confidence": "trusted",
                    "metadata": {},
                }
                for index in range(151)
            ],
            "edges": [
                {"id": f"edge_{index}", "source": f"node_{index}", "target": f"node_{index + 1}"}
                for index in range(150)
            ],
        }
    )

    monkeypatch.setattr(main_module, "trace_model_graph", lambda *_args, **_kwargs: huge_graph)

    response = client.post(
        "/api/inspect/resource/preview",
        files=[("files", ("resource.py", MANY_LAYER_RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["graph_diagnostics"]["strategy"] == "state_dict_grouped"
    assert payload["graph_diagnostics"]["node_count"] <= 4
    assert payload["graph"]["nodes"][1]["kind"] == "ModuleGroup"
    assert "exceeds the Ops budget" in payload["graph_diagnostics"]["note"]


def test_bounded_graph_caps_wide_top_level_modules() -> None:
    specs = [{"name": f"head{index}.weight", "shape": [4, 4]} for index in range(30)]

    graph = build_bounded_graph_from_tensor_specs(specs, max_nodes=24)

    assert len(graph.nodes) <= 24
    assert graph.nodes[1].metadata["aggregation_depth"] == 0


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


def test_resource_report_exports_shareable_markdown() -> None:
    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("resource.py", RGB_REPORT_RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py", "steps": "1"},
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    export = client.get(f"/api/runs/{run_id}/report/export.md")

    assert export.status_code == 200
    assert export.headers["content-type"].startswith("text/markdown")
    assert f'attachment; filename="{run_id}-report.md"' in export.headers["content-disposition"]
    markdown = export.text
    assert markdown.startswith(f"# PulseGraph Run Report: {run_id}")
    assert "Generated checkpoint: step 1" in markdown
    assert "run_kind: resource-training" in markdown
    assert "resource_name: rgb_report_resource" in markdown
    assert "## Layer Health" in markdown
    assert "## Error Analysis" in markdown
    assert "green -> red" in markdown or "blue -> red" in markdown


def test_resource_report_exports_printable_html() -> None:
    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("resource.py", RGB_REPORT_RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py", "steps": "1"},
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    export = client.get(f"/api/runs/{run_id}/report/export.html")

    assert export.status_code == 200
    assert export.headers["content-type"].startswith("text/html")
    html = export.text
    assert f"<title>PulseGraph Run Report: {run_id}</title>" in html
    assert "window.print()" in html
    assert "resource-training" in html
    assert "rgb_report_resource" in html
    assert "Layer Health" in html
    assert "Error Analysis" in html
    assert "green -&gt; red" in html or "blue -&gt; red" in html


def test_train_resource_endpoint_respects_telemetry_stride() -> None:
    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("resource.py", RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py", "steps": "6", "telemetry_stride": "3"},
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    detail = client.get(f"/api/runs/{run_id}/detail").json()

    assert [metric["step"] for metric in detail["metrics"]] == [1, 3, 6]
    assert detail["config"]["steps"] == 6
    assert detail["config"]["telemetry_stride"] == 3


def test_train_resource_endpoint_publishes_run_status_schema_and_lifecycle() -> None:
    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("resource.py", RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py", "steps": "3", "telemetry_stride": "2"},
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    events = main_module.run_store.load_events(run_id)
    status_events = [event for event in events if event.type == "run_status"]

    assert [event.payload["phase"] for event in status_events[:3]] == ["queued", "loading", "building"]
    assert "training" in [event.payload["phase"] for event in status_events]
    assert status_events[-1].payload["phase"] == "completed"
    first_training = next(event for event in status_events if event.payload["phase"] == "training")
    assert first_training.source == "training"
    assert first_training.payload["message"] == "Training step 1/3"
    assert first_training.payload["step"] == 1
    assert first_training.payload["total_steps"] == 3
    assert 0 < first_training.payload["progress"] <= 1
    assert "elapsed_sec" in first_training.payload
    assert "eta_sec" in first_training.payload


def test_train_resource_endpoint_aggregates_layer_snapshots_but_preserves_full_details() -> None:
    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("resource.py", MANY_LAYER_RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py", "steps": "1", "telemetry_stride": "1"},
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    events = main_module.run_store.load_events(run_id)
    layer_events = [event for event in events if event.type == "layer_snapshot"]

    assert len(layer_events) == 1
    payload = layer_events[0].payload
    assert payload["mode"] == "aggregate"
    assert payload["layer_count"] > payload["sampled_layer_count"]
    assert payload["sampled_layer_count"] <= main_module.MAX_LIVE_LAYER_SAMPLES
    saved_layers = main_module.run_store.load_layer_snapshot(run_id, 1)
    assert len(saved_layers) == payload["layer_count"]

    report = client.get(f"/api/runs/{run_id}/report").json()
    layer_ids = {item["layer_id"] for item in report["layer_health"]}
    assert "__aggregate__" not in layer_ids
    assert len(layer_ids) > payload["sampled_layer_count"]


def test_train_resource_endpoint_persists_failed_completion_and_config() -> None:
    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("resource.py", FAILING_RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py", "steps": "3", "telemetry_stride": "2"},
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    detail = client.get(f"/api/runs/{run_id}/detail").json()
    events = main_module.run_store.load_events(run_id)

    assert detail["config"]["training_status"] == "failed"
    assert detail["config"]["error"] == "resource exploded at step 2"
    assert any(event.type == "metric" and event.step == 1 for event in events)
    assert events[-1].type == "run_complete"
    assert events[-1].payload["status"] == "failed"
    assert events[-1].payload["error"] == "resource exploded at step 2"


def test_cancel_api_cancels_running_resource_job_end_to_end() -> None:
    run_id = f"cancel-{uuid.uuid4().hex[:10]}"
    main_module.run_store.save_source_files(
        run_id,
        [("resource.py", SLOW_RESOURCE_SOURCE.encode())],
        "resource.py",
        "TrainingResource",
    )
    source_path = main_module.run_store.source_path(run_id)
    assert source_path is not None
    source_root = main_module.run_store.run_dir(run_id) / "source"
    main_module.run_store.save_config(
        run_id,
        {
            "source": "training-resource",
            "run_kind": "resource-training",
            "resource_name": "slow_resource",
            "task": "classification",
            "inference_only": False,
            "training_status": "queued",
            "training_recipe": "resource-contract",
            "steps": 8,
            "telemetry_stride": 4,
            "entry_file": "resource.py",
            "entry_class": "TrainingResource",
            "weights": "training",
            "cancel_requested": False,
        },
    )
    thread = threading.Thread(
        target=run_resource_training_job,
        args=(run_id, source_path, source_root, 8, 4),
        kwargs={
            "run_store": main_module.run_store,
            "run_registry": main_module.run_registry,
            "training_task_controller": main_module.training_task_controller,
            "max_live_layer_samples": main_module.MAX_LIVE_LAYER_SAMPLES,
        },
        daemon=True,
    )
    thread.start()
    _wait_for_condition(
        lambda: any(
            event.type == "run_status" and event.payload.get("phase") == "training" and event.step >= 1
            for event in main_module.run_store.load_events(run_id)
        )
    )

    cancel = client.post(f"/api/runs/{run_id}/cancel")
    thread.join(timeout=5)

    assert cancel.status_code == 200
    assert cancel.json()["cancel_requested"] is True
    assert not thread.is_alive()
    detail = client.get(f"/api/runs/{run_id}/detail").json()
    events = main_module.run_store.load_events(run_id)
    assert detail["config"]["training_status"] == "cancelled"
    assert any(event.type == "run_status" and event.payload["phase"] == "cancelling" for event in events)
    assert any(event.type == "run_status" and event.payload["phase"] == "cancelled" for event in events)
    assert events[-1].type == "run_complete"
    assert events[-1].payload["status"] == "cancelled"


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
