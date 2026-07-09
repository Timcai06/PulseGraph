from __future__ import annotations

import json
from html import escape
from typing import Any

from app.schemas import ErrorAnalysis, RunDetail, RunReport


def _value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _class_name(label: int, error_analysis: ErrorAnalysis) -> str:
    if error_analysis.class_names and 0 <= label < len(error_analysis.class_names):
        return error_analysis.class_names[label]
    return str(label)


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_value(cell) for cell in row) + " |")
    return lines


def render_run_report_markdown(detail: RunDetail, report: RunReport) -> str:
    lines: list[str] = [
        f"# PulseGraph Run Report: {report.run_id}",
        "",
        f"Generated checkpoint: step {_value(report.generated_for_checkpoint)}",
        f"Completed: {_value(detail.completed)}",
        f"Events: {_value(detail.event_count)}",
        f"Checkpoints: {_value(len(detail.checkpoints))}",
        "",
        "## Summary",
        "",
        *_table(
            ["Metric", "Value"],
            [
                ["Final loss", report.final_loss],
                ["Best accuracy", report.best_accuracy],
                ["Overfit gap", report.overfit_gap],
                ["Plateau step", report.loss_plateau_step],
            ],
        ),
        "",
        "## Insights",
        "",
    ]

    for insight in report.insights:
        lines.append(f"- **{insight.severity.upper()}** {insight.title}: {insight.detail}")
        if insight.suggestion:
            lines.append(f"  - Suggestion: {insight.suggestion}")
    if not report.insights:
        lines.append("- No report insights were generated.")

    lines.extend(["", "## Config", ""])
    if detail.config:
        for key in sorted(detail.config):
            lines.append(f"- {key}: {_value(detail.config[key])}")
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
                lines.append(f"- sample {sample.get('index')}: {label} -> {prediction}")
        else:
            lines.append("No misclassified probe samples were recorded.")
    else:
        lines.append("No error analysis was available.")

    lines.extend(["", "## Provenance", ""])
    lines.append(f"- Source: {'recorded' if detail.source else 'missing'}")
    lines.append(f"- Graph: {'recorded' if detail.graph else 'missing'}")
    lines.append(f"- Probe samples: {'recorded' if detail.has_samples else 'missing'}")
    if detail.source_files:
        lines.append(f"- Source files: {', '.join(detail.source_files)}")

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
