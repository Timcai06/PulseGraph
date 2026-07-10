import textwrap

import torch
from fastapi.testclient import TestClient

from app.main import app, run_store


client = TestClient(app)


DETECTION_RESOURCE_SOURCE = textwrap.dedent(
    """
    import torch
    from torch import nn


    class TinyDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Conv2d(3, 2, kernel_size=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.box_head = nn.Linear(2, 4)
            self.score_head = nn.Linear(2, 1)
            with torch.no_grad():
                self.box_head.weight.zero_()
                self.box_head.bias.copy_(torch.tensor([1.0, 1.0, 6.0, 6.0]))
                self.score_head.weight.zero_()
                self.score_head.bias.fill_(1.5)

        def forward(self, images, targets=None):
            batch = torch.stack(images) if isinstance(images, list) else images
            features = self.pool(self.backbone(batch)).flatten(1)
            boxes = self.box_head(features)
            scores = torch.sigmoid(self.score_head(features)).squeeze(1)
            if targets is not None:
                target_boxes = torch.stack([target["boxes"][0].float() for target in targets]).to(batch.device)
                return {
                    "loss_box_reg": torch.abs(boxes - target_boxes).mean(),
                    "loss_classifier": torch.square(1.0 - scores).mean(),
                }
            return [
                {
                    "boxes": boxes[row : row + 1],
                    "labels": torch.tensor([1], device=batch.device),
                    "scores": scores[row : row + 1],
                }
                for row in range(batch.shape[0])
            ]


    def metadata():
        return {
            "name": "tiny_detection_resource",
            "task": "detection",
            "classes": 2,
            "class_names": ["background", "square"],
            "input_shape": [3, 8, 8],
            "output_schema": {"kind": "detection", "renderer": "box_overlay"},
        }


    def build_model():
        return TinyDetector()


    def train_batch(step, batch_size):
        images = torch.zeros(batch_size, 3, 8, 8)
        images[:, :, 1:7, 1:7] = 1.0
        targets = [
            {
                "boxes": torch.tensor([[1.0, 1.0, 6.0, 6.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            }
            for _ in range(batch_size)
        ]
        return images, targets


    def inference_sample(index):
        image = torch.zeros(3, 8, 8)
        image[:, 1:7, 1:7] = 1.0
        target = {
            "boxes": torch.tensor([[1.0, 1.0, 6.0, 6.0]]),
            "labels": torch.tensor([1]),
        }
        return image, target
    """
)


def test_detection_runtime_imports_and_serializes_preview_outputs() -> None:
    response = client.post(
        "/api/inspect/resource/preview",
        files=[("files", ("resource.py", DETECTION_RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["resource"]["task"] == "detection"
    assert payload["resource"]["output_schema"]["renderer"] == "box_overlay"
    assert payload["resource"]["metric_schema"]["primary"] == "mean_iou"
    assert payload["resource"]["metric_schema"]["monitors"] == ["loss", "mean_iou"]
    assert payload["samples"][0]["task"] == "detection"
    assert payload["samples"][0]["label"] is None
    assert payload["samples"][0]["output"] == {
        "kind": "detection",
        "boxes": [[1.0, 1.0, 6.0, 6.0]],
        "labels": [1],
        "scores": [],
        "label_names": ["square"],
        "total_detections": 1,
        "truncated": False,
    }


def test_detection_runtime_trains_replays_and_persists_structured_targets() -> None:
    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("resource.py", DETECTION_RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py", "steps": "2"},
    )

    assert response.status_code == 200
    payload = response.json()
    run_id = payload["run_id"]
    assert payload["resource"]["task"] == "detection"

    detail = client.get(f"/api/runs/{run_id}/detail").json()
    assert detail["config"]["task"] == "detection"
    assert detail["config"]["metric_schema"]["primary"] == "mean_iou"
    assert len(detail["metrics"]) == 2
    assert detail["metrics"][0]["phase"] == "train"
    assert detail["metrics"][0]["mean_iou"] == 1.0
    assert "loss_box_reg" in detail["metrics"][0]
    assert "loss_classifier" in detail["metrics"][0]
    assert "step_time_ms" not in detail["metrics"][0]
    event_types = {event.type for event in run_store.load_events(run_id)}
    assert {"graph", "metric", "infra", "layer_snapshot", "checkpoint", "run_complete"} <= event_types

    bundle = torch.load(run_store.samples_path(run_id), map_location="cpu", weights_only=True)
    assert list(bundle["images"].shape[1:]) == [3, 8, 8]
    assert bundle["sample_source"] == "probe"
    assert bundle["targets"][0]["boxes"].tolist() == [[1.0, 1.0, 6.0, 6.0]]
    assert bundle["targets"][0]["labels"].tolist() == [1]

    forward = client.get(f"/api/runs/{run_id}/forward?index=0").json()
    assert forward["task"] == "detection"
    assert forward["output_schema"]["renderer"] == "box_overlay"
    assert forward["metric_schema"]["primary"] == "mean_iou"
    assert forward["label"] is None
    assert forward["prediction"] is None
    assert forward["probabilities"] == []
    assert forward["output"]["kind"] == "detection"
    assert forward["output"]["labels"] == [1]
    assert forward["output"]["label_names"] == ["square"]
    assert len(forward["output"]["boxes"]) == 1
    assert len(forward["output"]["scores"]) == 1
    assert any(layer["layer_id"] == "backbone" for layer in forward["layers"])


def test_detection_runtime_supports_tensor_only_models_across_telemetry_and_replay() -> None:
    source = DETECTION_RESOURCE_SOURCE.replace(
        "batch = torch.stack(images) if isinstance(images, list) else images",
        'if isinstance(images, list):\n            raise TypeError("tensor input required")\n        batch = images',
    )
    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("resource.py", source.encode(), "text/x-python"))],
        data={"entry_file": "resource.py", "steps": "1", "telemetry_stride": "1"},
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    detail = client.get(f"/api/runs/{run_id}/detail").json()
    assert detail["metrics"][0]["mean_iou"] == 1.0
    forward = client.get(f"/api/runs/{run_id}/forward?index=0")
    assert forward.status_code == 200
    assert forward.json()["output"]["kind"] == "detection"


def test_detection_final_live_metric_and_report_use_the_same_probe_samples() -> None:
    source = DETECTION_RESOURCE_SOURCE.replace(
        "torch.tensor([[1.0, 1.0, 6.0, 6.0]], dtype=torch.float32)",
        "torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32)",
    )
    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("resource.py", source.encode(), "text/x-python"))],
        data={"entry_file": "resource.py", "steps": "1", "telemetry_stride": "1"},
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    detail = client.get(f"/api/runs/{run_id}/detail").json()
    report = client.get(f"/api/runs/{run_id}/report").json()

    assert detail["metrics"][-1]["mean_iou"] == report["detection_analysis"]["mean_iou"]


def test_unknown_task_fails_with_supported_capabilities() -> None:
    source = DETECTION_RESOURCE_SOURCE.replace('"task": "detection"', '"task": "segmentation"')
    response = client.post(
        "/api/inspect/resource/preview",
        files=[("files", ("resource.py", source.encode(), "text/x-python"))],
        data={"entry_file": "resource.py"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Unsupported CV task 'segmentation'. Supported tasks: classification, detection."
    )
