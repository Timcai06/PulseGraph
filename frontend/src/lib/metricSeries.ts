import type { MetricSchema } from "../api/types";
import type { MetricPoint } from "../hooks/useRunStream";

export type MetricSeriesTone = "amber" | "green" | "cyan" | "violet" | "red";

export type MetricSeriesSpec = {
  key: string;
  label: string;
  tone: MetricSeriesTone;
  yAxisIndex?: 0 | 1;
  area?: boolean;
};

export type MetricSeriesContext = {
  task?: string;
  metricSchema?: MetricSchema | null;
  group?: MetricGroup;
};

export type MetricGroup = "quality" | "optimization" | "infra";

export type PrimaryMetricSignal = {
  key: string;
  label: string;
  format: "float" | "percent";
};

const READABLE_LABELS: Record<string, string> = {
  accuracy: "Accuracy",
  learning_rate: "LR",
  loss: "Loss",
  loss_box_reg: "Loss box reg",
  loss_classifier: "Loss classifier",
  mean_iou: "Mean IoU",
  memory_peak_mb: "Memory MB",
  samples_per_sec: "Samples / sec",
  step_time_ms: "Step ms"
};

const METRIC_TONES: Record<string, MetricSeriesTone> = {
  loss: "amber",
  accuracy: "green",
  mean_iou: "green",
  samples_per_sec: "green",
  memory_peak_mb: "violet",
  step_time_ms: "cyan",
  loss_classifier: "violet",
  loss_box_reg: "red"
};

const CLASSIFICATION_SERIES = ["loss", "accuracy", "step_time_ms"] as const;
const DETECTION_SERIES = ["loss", "loss_classifier", "loss_box_reg", "mean_iou", "step_time_ms"] as const;
const HIDDEN_FALLBACK_KEYS = new Set(["elapsed_sec", "learning_rate", "memory_peak_mb", "samples_per_sec"]);

function metricValueFromShortcuts(point: MetricPoint, key: string): number | undefined {
  switch (key) {
    case "loss":
      return point.loss;
    case "accuracy":
      return point.accuracy;
    case "learning_rate":
      return point.learningRate;
    case "step_time_ms":
      return point.stepTimeMs;
    case "memory_peak_mb":
      return point.memoryPeakMb;
    default:
      return undefined;
  }
}

export function metricValue(point: MetricPoint, key: string): number | undefined {
  return point.values?.[key] ?? metricValueFromShortcuts(point, key);
}

function hasMetric(points: MetricPoint[], key: string): boolean {
  return points.some((point) => metricValue(point, key) != null);
}

function uniqueKeys(keys: Array<string | undefined>): string[] {
  const ordered: string[] = [];
  const seen = new Set<string>();
  for (const key of keys) {
    if (!key || seen.has(key)) continue;
    seen.add(key);
    ordered.push(key);
  }
  return ordered;
}

function metricLabel(key: string): string {
  if (READABLE_LABELS[key]) return READABLE_LABELS[key];
  return key
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function signalLabel(key: string): string {
  if (key === "accuracy") return "acc";
  if (key === "mean_iou") return "mean IoU";
  return metricLabel(key);
}

function metricFormat(key: string): PrimaryMetricSignal["format"] {
  return key === "accuracy" || key === "mean_iou" ? "percent" : "float";
}

function metricTone(key: string): MetricSeriesTone {
  return METRIC_TONES[key] ?? "violet";
}

function metricAxis(key: string): 0 | 1 {
  return key === "accuracy" || key === "mean_iou" ? 1 : 0;
}

function metricArea(key: string): boolean {
  return key === "loss" || key === "accuracy" || key === "mean_iou";
}

function normalizedTask(task?: string): string | undefined {
  const value = task?.trim().toLowerCase();
  return value ? value : undefined;
}

function inferredTask(points: MetricPoint[], context: MetricSeriesContext): string | undefined {
  const task = normalizedTask(context.task);
  if (task) return task;
  if (hasMetric(points, "mean_iou") || hasMetric(points, "loss_classifier") || hasMetric(points, "loss_box_reg")) {
    return "detection";
  }
  if (hasMetric(points, "accuracy")) return "classification";
  return undefined;
}

function genericSeriesKeys(points: MetricPoint[], context: MetricSeriesContext): string[] {
  const available = uniqueKeys(points.flatMap((point) => Object.keys(point.values ?? {})));
  const primary = typeof context.metricSchema?.primary === "string" ? context.metricSchema.primary : undefined;
  const monitors = Array.isArray(context.metricSchema?.monitors)
    ? context.metricSchema.monitors.filter((value): value is string => typeof value === "string")
    : [];
  const preferred = uniqueKeys(["loss", primary, ...monitors, "accuracy", "mean_iou"]);
  const remainder = available.filter((key) => !preferred.includes(key) && !HIDDEN_FALLBACK_KEYS.has(key) && key !== "step_time_ms");
  const selected = [...preferred.filter((key) => hasMetric(points, key)), ...remainder].slice(0, 4);
  return hasMetric(points, "step_time_ms") ? [...selected, "step_time_ms"] : selected;
}

export function deriveMetricSeries(points: MetricPoint[], context: MetricSeriesContext = {}): MetricSeriesSpec[] {
  const task = inferredTask(points, context);
  const keys =
    task === "classification"
      ? CLASSIFICATION_SERIES.filter((key) => hasMetric(points, key))
      : task === "detection"
        ? DETECTION_SERIES.filter((key) => hasMetric(points, key))
        : genericSeriesKeys(points, context);

  const groupedKeys = context.group === "quality"
    ? keys.filter((key) => key === "accuracy" || key === "mean_iou")
    : context.group === "optimization"
      ? keys.filter((key) => key === "loss" || key.startsWith("loss_") || key === "learning_rate")
      : context.group === "infra"
        ? ["step_time_ms", "samples_per_sec", "memory_peak_mb"].filter((key) => hasMetric(points, key))
        : keys;

  const visibleKeys = context.group ? groupedKeys : keys;
  return visibleKeys.slice(0, 5).map((key) => ({
    key,
    label: metricLabel(key),
    tone: metricTone(key),
    yAxisIndex: metricAxis(key),
    area: metricArea(key)
  }));
}

export function latestMetricValue(points: MetricPoint[], key: string): number | undefined {
  for (let index = points.length - 1; index >= 0; index -= 1) {
    const value = metricValue(points[index], key);
    if (value != null) return value;
  }
  return undefined;
}

export function derivePrimaryMetricSignal(points: MetricPoint[], context: MetricSeriesContext = {}): PrimaryMetricSignal | undefined {
  const task = inferredTask(points, context);
  const primary = typeof context.metricSchema?.primary === "string" ? context.metricSchema.primary : undefined;
  const candidates =
    task === "classification"
      ? ["accuracy", primary]
      : task === "detection"
        ? ["mean_iou", primary, "accuracy"]
        : [primary, "accuracy", "mean_iou"];

  const key = uniqueKeys(candidates).find((candidate) => candidate && hasMetric(points, candidate));
  if (!key) return undefined;
  return {
    key,
    label: signalLabel(key),
    format: metricFormat(key)
  };
}
