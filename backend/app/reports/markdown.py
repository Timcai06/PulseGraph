from __future__ import annotations

import json
from html import escape
from typing import Any

from app.schemas import DetectionAnalysis, ErrorAnalysis, RunDetail, RunReport


def _value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _markdown_text(value: Any) -> str:
    text = _value(value).replace("\r", " ").replace("\n", " ")
    for character in "\\`[]()<!|":
        text = text.replace(character, f"\\{character}")
    return text


def _class_name(label: int, error_analysis: ErrorAnalysis) -> str:
    if error_analysis.class_names and 0 <= label < len(error_analysis.class_names):
        return error_analysis.class_names[label]
    return str(label)


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(_markdown_text(header) for header in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_text(cell) for cell in row) + " |")
    return lines


def _format_box(box: list[float]) -> str:
    return "[" + ", ".join(f"{coordinate:.2f}" for coordinate in box) + "]"


def _render_detection_evidence(analysis: DetectionAnalysis) -> list[str]:
    lines: list[str] = ["## Detection Evidence", ""]
    if analysis.mean_iou is not None:
        lines.append(f"- Mean IoU: {_value(analysis.mean_iou)}")
    lines.append(f"- Evaluated samples: {analysis.evaluated_samples}")
    lines.append("")
    if not analysis.evidence:
        lines.append("No detection evidence was available.")
        return lines

    for sample in analysis.evidence:
        lines.append(f"### sample {sample.sample_index}")
        lines.append("")
        lines.append(f"- Sample mean IoU: {_value(sample.mean_iou)}")
        lines.append(f"- Image shape: {_value(sample.image_shape)}")
        if sample.predicted:
            prediction_count = (
                f"{len(sample.predicted)} shown of {sample.predicted_total}"
                if sample.predicted_truncated
                else str(sample.predicted_total)
            )
            lines.append(f"- Predicted boxes ({prediction_count}):")
            for box in sample.predicted:
                score = f", score={box.score:.4g}" if box.score is not None else ""
                lines.append(
                    f"  - {_markdown_text(box.label_name or box.label)}: {_format_box(box.box)}"
                    f"{score}, matched IoU={_value(box.matched_iou)}"
                )
        else:
            lines.append("- Predicted boxes: none")
        if sample.target:
            target_count = (
                f"{len(sample.target)} shown of {sample.target_total}"
                if sample.target_truncated
                else str(sample.target_total)
            )
            lines.append(f"- Target boxes ({target_count}):")
            for box in sample.target:
                lines.append(
                    f"  - {_markdown_text(box.label_name or box.label)}: {_format_box(box.box)}, matched IoU={_value(box.matched_iou)}"
                )
        else:
            lines.append("- Target boxes: none")
        lines.append("")
    return lines


def _summary_rows(report: RunReport) -> list[list[Any]]:
    rows: list[list[Any]] = [["Final loss", report.final_loss]]
    if report.task == "classification":
        rows.extend(
            [
                ["Best accuracy", report.best_accuracy],
                ["Overfit gap", report.overfit_gap],
            ]
        )
    rows.append(["Plateau step", report.loss_plateau_step])
    return rows


def render_run_report_markdown(detail: RunDetail, report: RunReport) -> str:
    lines: list[str] = [
        f"# PulseGraph Run Report: {_markdown_text(report.run_id)}",
        "",
        f"Task: {_markdown_text(report.task)}",
        f"Generated checkpoint: step {_value(report.generated_for_checkpoint)}",
        f"Completed: {_value(detail.completed)}",
        f"Events: {_value(detail.event_count)}",
        f"Checkpoints: {_value(len(detail.checkpoints))}",
        "",
        "## Summary",
        "",
        *_table(["Metric", "Value"], _summary_rows(report)),
        "",
        "## Task Metrics",
        "",
    ]

    if report.task_metrics:
        lines.extend(_table(["Metric", "Value"], [[item.label, item.value] for item in report.task_metrics]))
    else:
        lines.append("No task-specific metrics were available.")

    lines.extend(["", "## Insights", ""])
    for insight in report.insights:
        lines.append(
            f"- **{_markdown_text(insight.severity.upper())}** "
            f"{_markdown_text(insight.title)}: {_markdown_text(insight.detail)}"
        )
        if insight.suggestion:
            lines.append(f"  - Suggestion: {_markdown_text(insight.suggestion)}")
    if not report.insights:
        lines.append("- No report insights were generated.")

    lines.extend(["", "## Config", ""])
    if detail.config:
        for key in sorted(detail.config):
            lines.append(f"- {_markdown_text(key)}: {_markdown_text(detail.config[key])}")
    else:
        lines.append("- No config was recorded.")

    lines.extend(["", "## Layer Health", ""])
    if report.layer_health:
        lines.extend(
            _table(
                ["Layer", "Sparsity", "Grad norm", "Trend", "Weight drift"],
                [
                    [
                        layer.layer_id,
                        layer.mean_sparsity,
                        layer.last_gradient_norm,
                        layer.gradient_trend,
                        layer.weight_std_drift,
                    ]
                    for layer in report.layer_health
                ],
            )
        )
    else:
        lines.append("No layer health snapshots were recorded.")

    if report.task == "classification":
        lines.extend(["", "## Checkpoint Evaluation", ""])
        if report.checkpoint_evaluations:
            lines.extend(
                _table(
                    ["Step", "Accuracy", "Sample count"],
                    [[item.step, item.accuracy, item.sample_count] for item in report.checkpoint_evaluations],
                )
            )
        else:
            lines.append("No checkpoint evaluations were available.")

        lines.extend(["", "## Error Analysis", ""])
        if report.error_analysis and report.error_analysis.labels:
            analysis = report.error_analysis
            headers = ["true / pred", *[_class_name(label, analysis) for label in analysis.labels]]
            rows = [
                [_class_name(analysis.labels[row_index], analysis), *row]
                for row_index, row in enumerate(analysis.confusion)
            ]
            lines.extend(_table(headers, rows))
            lines.append("")
            if analysis.misclassified:
                lines.append("### Misclassified Samples")
                lines.append("")
                for sample in analysis.misclassified:
                    label = sample.get("label_name") or _class_name(int(sample.get("label", 0)), analysis)
                    prediction = sample.get("prediction_name") or _class_name(int(sample.get("prediction", 0)), analysis)
                    lines.append(
                        f"- sample {_markdown_text(sample.get('index'))}: "
                        f"{_markdown_text(label)} -> {_markdown_text(prediction)}"
                    )
            else:
                lines.append("No misclassified probe samples were recorded.")
        else:
            lines.append("No error analysis was available.")

    if report.detection_analysis is not None:
        lines.extend(["", *_render_detection_evidence(report.detection_analysis)])

    lines.extend(["", "## Provenance", ""])
    lines.append(f"- Source: {'recorded' if detail.source else 'missing'}")
    lines.append(f"- Graph: {'recorded' if detail.graph else 'missing'}")
    lines.append(f"- Probe samples: {'recorded' if detail.has_samples else 'missing'}")
    if detail.source_files:
        lines.append(f"- Source files: {', '.join(_markdown_text(name) for name in detail.source_files)}")

    return "\n".join(lines).rstrip() + "\n"


def render_run_report_html(detail: RunDetail, report: RunReport) -> str:
    title = f"PulseGraph Run Report: {report.run_id}"
    markdown = render_run_report_markdown(detail, report)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    body {{
      margin: 0;
      color: #111827;
      background: #f8fafc;
    }}
    main {{
      max-width: 960px;
      margin: 0 auto;
      padding: 40px 28px 64px;
    }}
    .toolbar {{
      position: sticky;
      top: 0;
      display: flex;
      justify-content: flex-end;
      padding: 12px 0;
      background: rgba(248, 250, 252, 0.92);
      backdrop-filter: blur(10px);
    }}
    button {{
      border: 1px solid #0f172a;
      border-radius: 6px;
      padding: 8px 12px;
      color: #fff;
      background: #0f172a;
      font: inherit;
      cursor: pointer;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 24px;
      background: #fff;
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
      font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
      line-height: 1.65;
    }}
    @media print {{
      body {{ background: #fff; }}
      main {{ max-width: none; padding: 0; }}
      .toolbar {{ display: none; }}
      pre {{ border: 0; box-shadow: none; padding: 0; }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="toolbar"><button type="button" onclick="window.print()">Print / Save PDF</button></div>
    <pre>{escape(markdown)}</pre>
  </main>
</body>
</html>
"""
