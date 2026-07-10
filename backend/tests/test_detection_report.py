import io

import pytest
import torch

from app.events.run_store import RunStore
from app.reports.analyzer import build_detection_analysis, build_run_report
from app.reports.markdown import render_run_report_html, render_run_report_markdown
from app.runtime.detection import DetectionContractError, mean_detection_iou
from app.runtime.replay import build_run_detail


DETECTION_SOURCE = """import torch
from torch import nn


class TinyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.boxes = nn.Parameter(
            torch.tensor(
                [
                    [1.0, 1.0, 6.0, 6.0],
                    [2.0, 2.0, 7.0, 7.0],
                ],
                dtype=torch.float32,
            )
        )
        self.scores = nn.Parameter(torch.tensor([0.95, 0.85], dtype=torch.float32))

    def forward(self, images):
        batch_size = len(images) if isinstance(images, list) else images.shape[0]
        outputs = []
        for index in range(batch_size):
            slot = index % self.boxes.shape[0]
            outputs.append(
                {
                    "boxes": self.boxes[slot].unsqueeze(0),
                    "labels": torch.tensor([1]),
                    "scores": self.scores[slot].unsqueeze(0),
                }
            )
        return outputs
"""


def _register_detection_run(run_id: str) -> RunStore:
    store = RunStore()
    store.save_source(run_id, DETECTION_SOURCE, entry_class="TinyDetector")
    store.save_config(
        run_id,
        {
            "source": "recorded-python",
            "run_kind": "source-import",
            "task": "detection",
            "output_schema": {"kind": "detection", "renderer": "box_overlay"},
            "class_names": ["background", "square"],
            "weights": "trained",
        },
    )

    model_namespace: dict[str, object] = {}
    exec(DETECTION_SOURCE, model_namespace)
    model = model_namespace["TinyDetector"]()

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    store.save_checkpoint_bytes(run_id, 12, checkpoint.getvalue(), epoch=1)

    images = torch.zeros(2, 3, 8, 8)
    images[0, :, 1:7, 1:7] = 1.0
    images[1, :, 1:7, 1:7] = 0.7
    sample = io.BytesIO()
    torch.save(
        {
            "images": images,
            "targets": [
                {"boxes": torch.tensor([[1.0, 1.0, 6.0, 6.0]]), "labels": torch.tensor([1])},
                {"boxes": torch.tensor([[1.0, 1.0, 7.0, 7.0]]), "labels": torch.tensor([1])},
            ],
            "sample_source": "probe",
        },
        sample,
    )
    store.save_samples(run_id, sample.getvalue())
    return store


def test_detection_analysis_helper_consumes_structured_dicts() -> None:
    images = torch.zeros(1, 3, 8, 8)
    images[0, :, 1:7, 1:7] = 1.0

    analysis = build_detection_analysis(
        images=images,
        predictions=[
            {
                "boxes": [[1.0, 1.0, 6.0, 6.0], [0.0, 0.0, 2.0, 2.0]],
                "labels": [1, 0],
                "scores": [0.95, 0.2],
            }
        ],
        targets=[{"boxes": [[1.0, 1.0, 6.0, 6.0]], "labels": [1]}],
        class_names=["background", "square"],
    )

    assert analysis.evaluated_samples == 1
    assert analysis.mean_iou == 1.0
    assert analysis.evidence[0].predicted[0].label_name == "square"
    assert analysis.evidence[0].predicted[0].matched_iou == 1.0
    assert analysis.evidence[0].target[0].matched_iou == 1.0


def test_detection_matching_is_one_to_one_in_live_metrics_and_reports() -> None:
    images = torch.zeros(1, 3, 8, 8)
    predictions = [{"boxes": [[1.0, 1.0, 6.0, 6.0]], "labels": [1], "scores": [0.9]}]
    targets = [{"boxes": [[1.0, 1.0, 6.0, 6.0], [1.0, 1.0, 6.0, 6.0]], "labels": [1, 1]}]

    live_iou = mean_detection_iou(predictions, targets, expected_size=1)
    report = build_detection_analysis(images, predictions, targets)

    assert live_iou == 0.5
    assert report.mean_iou == 0.5
    assert report.evidence[0].target[0].matched_iou == 1.0
    assert report.evidence[0].target[1].matched_iou is None


def test_detection_analysis_rejects_one_dict_for_a_multi_image_batch() -> None:
    images = torch.zeros(2, 3, 8, 8)
    prediction = {"boxes": [[1.0, 1.0, 6.0, 6.0]], "labels": [1]}
    targets = [prediction, prediction]

    with pytest.raises(DetectionContractError, match="one detection dict for a batch of 2"):
        build_detection_analysis(images, prediction, targets)


def test_detection_evidence_bounds_images_and_box_counts() -> None:
    images = torch.zeros(1, 3, 320, 640)
    predictions = [{
        "boxes": [[0.0, 0.0, 10.0, 10.0]] * 140,
        "labels": [1] * 140,
        "scores": [0.8] * 140,
    }]
    targets = [{"boxes": [[0.0, 0.0, 10.0, 10.0]] * 130, "labels": [1] * 130}]

    analysis = build_detection_analysis(images, predictions, targets, class_names=["background", "square"])
    sample = analysis.evidence[0]

    assert sample.image_shape == [3, 80, 160]
    assert len(sample.image_pixels) == 3 * 80 * 160
    assert len(sample.predicted) == 100
    assert len(sample.target) == 100
    assert sample.predicted_total == 140
    assert sample.target_total == 130
    assert sample.predicted_truncated is True
    assert sample.target_truncated is True
    assert sample.predicted[0].box == [0.0, 0.0, 2.5, 2.5]


def test_detection_report_surfaces_mean_iou_and_exportable_evidence() -> None:
    run_id = "detection-report"
    store = _register_detection_run(run_id)

    detail = build_run_detail(store, run_id)
    assert detail is not None

    report = build_run_report(store, detail)

    assert report.task == "detection"
    assert report.generated_for_checkpoint == 12
    assert report.detection_analysis is not None
    assert report.detection_analysis.evaluated_samples == 2
    assert report.detection_analysis.mean_iou == 0.8472
    assert report.task_metrics[0].key == "mean_iou"
    assert report.task_metrics[0].value == 0.8472
    assert report.task_metrics[1].key == "evaluated_samples"
    assert report.task_metrics[1].value == 2

    sample = report.detection_analysis.evidence[1]
    assert sample.sample_index == 1
    assert sample.mean_iou == 0.6944
    assert sample.predicted[0].score == 0.85
    assert sample.predicted[0].box == [2.0, 2.0, 7.0, 7.0]
    assert sample.target[0].box == [1.0, 1.0, 7.0, 7.0]

    markdown = render_run_report_markdown(detail, report)
    html = render_run_report_html(detail, report)

    assert "Task: detection" in markdown
    assert "## Task Metrics" in markdown
    assert "Mean IoU" in markdown
    assert "## Detection Evidence" in markdown
    assert "sample 1" in markdown
    assert "0.85" in markdown
    assert "Best accuracy" not in markdown
    assert "## Error Analysis" not in markdown
    assert "Detection Evidence" in html
    assert "Task Metrics" in html
