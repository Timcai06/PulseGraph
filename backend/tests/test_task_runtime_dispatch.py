import textwrap

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


DETECTION_RESOURCE_SOURCE = textwrap.dedent(
    """
    import torch
    from torch import nn


    class TinyDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.score = nn.Parameter(torch.tensor(0.8))

        def forward(self, images):
            batch_size = len(images) if isinstance(images, list) else images.shape[0]
            return [
                {
                    "boxes": torch.tensor([[1.0, 1.0, 6.0, 6.0]]),
                    "labels": torch.tensor([1]),
                    "scores": self.score.expand(1),
                }
                for _ in range(batch_size)
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
    assert payload["resource"]["metric_schema"]["primary"] == "loss"
    assert payload["samples"][0]["task"] == "detection"
    assert payload["samples"][0]["label"] is None
    assert payload["samples"][0]["output"] == {
        "kind": "detection",
        "boxes": [[1.0, 1.0, 6.0, 6.0]],
        "labels": [1],
        "scores": [],
        "label_names": ["square"],
    }


def test_detection_runtime_rejects_training_with_capability_error() -> None:
    response = client.post(
        "/api/runs/train-resource",
        files=[("files", ("resource.py", DETECTION_RESOURCE_SOURCE.encode(), "text/x-python"))],
        data={"entry_file": "resource.py", "steps": "1"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Task 'detection' supports resource import and preview, but training is not available in this runtime yet."
    )


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
