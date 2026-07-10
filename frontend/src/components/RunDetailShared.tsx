import type { GraphNode, RunDetail, RunReport } from "../api/client";
import { displayClassName, normalizeImageShape, type DetectionView } from "../lib/inferenceView";
import { ImagePreview } from "./ImagePreview";

export type SummaryCard = {
  key: string;
  label: string;
  value: string;
  tone?: "live" | "success" | "muted";
};

export type TimelineEntry = {
  key: string;
  label: string;
  detail: string;
  step?: number;
  meta?: string;
  tone: "metric" | "checkpoint" | "system";
};

export type DetectionEvidenceSample = NonNullable<NonNullable<RunReport["detection_analysis"]>["evidence"]>[number];

const METRIC_KEY_PRIORITY = [
  "loss",
  "accuracy",
  "mean_iou",
  "learning_rate",
  "samples_per_sec",
  "step_time_ms",
  "memory_peak_mb"
] as const;

export function severityClass(severity: string) {
  return severity === "critical" ? "critical" : severity === "warning" ? "warning" : "info";
}

export function formatMetricValue(value: number | string | null | undefined, unit?: string | null) {
  if (value == null) return "–";
  const rendered =
    typeof value === "number"
      ? Number.isInteger(value)
        ? String(value)
        : value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "")
      : value;
  return unit ? `${rendered} ${unit}` : rendered;
}

export function formatBox(box: number[]) {
  return box.map((value) => value.toFixed(2).replace(/\.00$/, "")).join(", ");
}

export function detectionLabel(label: number, labelName?: string | null) {
  return labelName || String(label);
}

export function evidenceDetection(sample: DetectionEvidenceSample): DetectionView {
  const totalCount = sample.predicted_total ?? sample.predicted.length;
  return {
    boxes: sample.predicted.flatMap((box, index) =>
      box.box.length === 4
        ? [
            {
              index,
              label: box.label,
              labelName: detectionLabel(box.label, box.label_name),
              score: box.score ?? undefined,
              coordinates: [box.box[0], box.box[1], box.box[2], box.box[3]] as [number, number, number, number]
            }
          ]
        : []
    ),
    totalCount,
    truncated: sample.predicted_truncated === true || totalCount > sample.predicted.length
  };
}

export function asNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

export function asMetricText(value: unknown) {
  if (typeof value === "number") return formatMetricValue(value);
  if (typeof value === "string" && value.trim().length > 0) return value;
  return undefined;
}

export function formatDateTime(timestampSeconds: number | undefined) {
  if (!timestampSeconds || !Number.isFinite(timestampSeconds)) return "–";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(timestampSeconds * 1000));
}

export function formatClock(timestampMs: number | undefined) {
  if (!timestampMs || !Number.isFinite(timestampMs)) return "–";
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  }).format(new Date(timestampMs));
}

export function formatShape(shape?: number[] | null) {
  return shape?.length ? shape.join(" × ") : "–";
}

export function formatSourceOrigin(detail: RunDetail | undefined) {
  if (!detail?.source) return "missing";
  return detail.source_origin === "user-attached" ? "attached" : "recorded";
}

export function taskName(detail: RunDetail | undefined, report: RunReport | undefined) {
  const reportTask = report?.task;
  if (typeof reportTask === "string" && reportTask.trim()) return reportTask;
  const detailTask = detail?.config?.task;
  if (typeof detailTask === "string" && detailTask.trim()) return detailTask.toLowerCase();
  return "classification";
}

export function latestMetricRow(detail: RunDetail | undefined) {
  const metrics = detail?.metrics;
  return metrics?.[metrics.length - 1];
}

export function latestStep(detail: RunDetail | undefined) {
  const metricStep = asNumber(latestMetricRow(detail)?.step);
  if (metricStep != null) return metricStep;
  const checkpoints = detail?.checkpoints;
  return checkpoints?.[checkpoints.length - 1]?.step ?? 0;
}

export function metricRecordKeys(metrics: RunDetail["metrics"]) {
  const keys = new Set<string>();
  for (const metric of metrics) {
    for (const [key, value] of Object.entries(metric)) {
      if (key === "step" || key === "epoch" || value == null) continue;
      if (typeof value === "number" || typeof value === "string") keys.add(key);
    }
  }
  return [...keys].sort((left, right) => {
    const leftIndex = METRIC_KEY_PRIORITY.indexOf(left as (typeof METRIC_KEY_PRIORITY)[number]);
    const rightIndex = METRIC_KEY_PRIORITY.indexOf(right as (typeof METRIC_KEY_PRIORITY)[number]);
    if (leftIndex !== -1 || rightIndex !== -1) {
      return (leftIndex === -1 ? Number.MAX_SAFE_INTEGER : leftIndex) - (rightIndex === -1 ? Number.MAX_SAFE_INTEGER : rightIndex);
    }
    return left.localeCompare(right);
  });
}

export function metricHighlights(detail: RunDetail | undefined) {
  const latest = latestMetricRow(detail);
  if (!latest) return [];
  return metricRecordKeys(detail?.metrics ?? [])
    .slice(0, 6)
    .flatMap((key) => {
      const text = asMetricText(latest[key]);
      return text ? [{ key, label: key.replace(/_/g, " "), value: text }] : [];
    });
}

export function shouldPollRunDetail(detail: RunDetail | undefined) {
  return Boolean(detail && !detail.completed);
}

export function buildRunTimeline(detail: RunDetail): TimelineEntry[] {
  const latestMetrics = detail.metrics
    .slice(-16)
    .reverse()
    .map((metric, index) => {
      const focus = metricRecordKeys([metric])
        .slice(0, 3)
        .flatMap((key) => {
          const text = asMetricText(metric[key]);
          return text ? [`${key.replace(/_/g, " ")} ${text}`] : [];
        })
        .join(" · ");
      return {
        key: `metric-${asNumber(metric.step) ?? index}-${index}`,
        label: "Metric snapshot",
        detail: focus || "Recorded metric payload",
        step: asNumber(metric.step),
        meta: metric.epoch != null ? `epoch ${metric.epoch}` : undefined,
        tone: "metric" as const
      };
    });

  const checkpoints = detail.checkpoints
    .slice()
    .reverse()
    .map((checkpoint) => ({
      key: `checkpoint-${checkpoint.step}`,
      label: "Checkpoint saved",
      detail: `${formatMetricValue(checkpoint.size_mb, "MB")}${checkpoint.fingerprint ? ` · ${checkpoint.fingerprint.slice(0, 10)}` : ""}`,
      step: checkpoint.step,
      meta: checkpoint.path,
      tone: "checkpoint" as const
    }));

  const system: TimelineEntry[] = [
    {
      key: "created",
      label: "Run created",
      detail: `${detail.run_id} opened for tracking`,
      meta: formatDateTime(detail.created_at),
      tone: "system"
    }
  ];

  if (detail.completed) {
    system.unshift({
      key: "completed",
      label: "Run completed",
      detail: "Live polling stopped after the final recorded detail snapshot.",
      step: latestStep(detail),
      tone: "system"
    });
  }

  return [...system, ...checkpoints, ...latestMetrics].slice(0, 24);
}

export function overviewCards(detail: RunDetail | undefined, report: RunReport | undefined, lastRefreshAt: number | undefined): SummaryCard[] {
  if (!detail) return [];
  const latestCheckpoint = detail.checkpoints[detail.checkpoints.length - 1];
  const task = taskName(detail, report);
  const cards: SummaryCard[] = [
    {
      key: "status",
      label: "status",
      value: detail.completed ? "completed" : "live sync",
      tone: detail.completed ? "success" : "live"
    },
    {
      key: "step",
      label: "current step",
      value: String(latestStep(detail))
    },
    {
      key: "checkpoint",
      label: "latest checkpoint",
      value: latestCheckpoint ? `step ${latestCheckpoint.step}` : "not saved yet"
    },
    {
      key: "task",
      label: "task",
      value: task
    },
    {
      key: "provenance",
      label: "source provenance",
      value: formatSourceOrigin(detail)
    },
    {
      key: "refresh",
      label: "last refresh",
      value: formatClock(lastRefreshAt),
      tone: "muted"
    }
  ];

  const highlight = metricHighlights(detail).slice(0, 2);
  highlight.forEach((card) => cards.push(card));
  return cards;
}

export function reportSummaryCards(detail: RunDetail | undefined, report: RunReport | undefined) {
  if (!detail || !report) return [];
  const reportTask = taskName(detail, report);
  const cards: SummaryCard[] = [{ key: "task", label: "task", value: reportTask }];
  const seen = new Set(cards.map((card) => card.key));
  const addCard = (key: string, label: string, value: number | string | null | undefined, unit?: string | null) => {
    if (value == null || seen.has(key)) return;
    cards.push({ key, label, value: formatMetricValue(value, unit) });
    seen.add(key);
  };

  report.task_metrics.forEach((metric) => addCard(metric.key, metric.label, metric.value, metric.unit));
  addCard("final_loss", "final loss", report.final_loss);
  if (reportTask === "classification") {
    addCard("best_accuracy", "best accuracy", report.best_accuracy);
    addCard("overfit_gap", "overfit gap", report.overfit_gap);
  }
  addCard("loss_plateau_step", "plateau step", report.loss_plateau_step);
  addCard("generated_for_checkpoint", "report checkpoint", report.generated_for_checkpoint);
  return cards;
}

export function formatConfidence(confidence: GraphNode["confidence"]) {
  return confidence.replace(/_/g, " ");
}

export function DetectionEvidenceCard({ sample }: { sample: DetectionEvidenceSample }) {
  const shape = normalizeImageShape(sample.image_shape, sample.image_pixels.length);
  const imageLabel = shape ? `${shape[2]}×${shape[1]}` : "image";
  const detection = evidenceDetection(sample);

  return (
    <article className="detection-evidence-card">
      <div className="detection-evidence-header">
        <strong>sample {sample.sample_index}</strong>
        <span>mean IoU {formatMetricValue(sample.mean_iou)}</span>
      </div>
      <div className="detection-evidence-body">
        <div className="detection-evidence-preview">
          <ImagePreview
            pixels={sample.image_pixels}
            imageShape={sample.image_shape}
            size="normal"
            detection={detection}
            overlayLabel={`Checkpoint detections for sample ${sample.sample_index}`}
          />
          <span className="detection-evidence-shape">{imageLabel}</span>
        </div>
        <div className="detection-evidence-columns">
          <div>
            <h4>
              Predicted
              {sample.predicted_truncated ? ` ${sample.predicted.length}/${sample.predicted_total}` : ""}
            </h4>
            {sample.predicted.length > 0 ? (
              <table className="report-table detection-box-table">
                <thead>
                  <tr>
                    <th>label</th>
                    <th>score</th>
                    <th>IoU</th>
                    <th>box</th>
                  </tr>
                </thead>
                <tbody>
                  {sample.predicted.map((box, index) => (
                    <tr key={`${sample.sample_index}-pred-${index}`}>
                      <td>{detectionLabel(box.label, box.label_name)}</td>
                      <td>{formatMetricValue(box.score)}</td>
                      <td>{formatMetricValue(box.matched_iou)}</td>
                      <td>{formatBox(box.box)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="empty-hint">No predicted boxes.</p>
            )}
          </div>
          <div>
            <h4>
              Target
              {sample.target_truncated ? ` ${sample.target.length}/${sample.target_total}` : ""}
            </h4>
            {sample.target.length > 0 ? (
              <table className="report-table detection-box-table">
                <thead>
                  <tr>
                    <th>label</th>
                    <th>IoU</th>
                    <th>box</th>
                  </tr>
                </thead>
                <tbody>
                  {sample.target.map((box, index) => (
                    <tr key={`${sample.sample_index}-target-${index}`}>
                      <td>{detectionLabel(box.label, box.label_name)}</td>
                      <td>{formatMetricValue(box.matched_iou)}</td>
                      <td>{formatBox(box.box)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="empty-hint">No target boxes.</p>
            )}
          </div>
        </div>
      </div>
    </article>
  );
}

export { displayClassName };
