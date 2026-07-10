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
  map50: "AP@0.50",
  mean_iou: "Mean IoU",
  memory_peak_mb: "Memory MB",
  samples_per_sec: "Samples / sec",
  step_time_ms: "Step ms",
  precision50: "Precision@0.50",
  recall50: "Recall@0.50"
};

const METRIC_TONES: Record<string, MetricSeriesTone> = {
  loss: "amber",
  accuracy: "green",
  map50: "green",
  mean_iou: "green",
  precision50: "cyan",
  recall50: "violet",
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

function isRatioMetric(key: string): boolean {
  return key === "accuracy" || key === "mean_iou" || key === "map50" || key === "precision50" || key === "recall50";
}

function metricFormat(key: string): PrimaryMetricSignal["format"] {
  return isRatioMetric(key) ? "percent" : "float";
}

function metricTone(key: string): MetricSeriesTone {
  return METRIC_TONES[key] ?? "violet";
}

function metricAxis(key: string): 0 | 1 {
  return isRatioMetric(key) ? 1 : 0;
}

function metricArea(key: string): boolean {
  return key === "loss" || isRatioMetric(key);
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

function schemaMonitors(context: MetricSeriesContext): string[] {
  return Array.isArray(context.metricSchema?.monitors)
    ? context.metricSchema.monitors.filter((value): value is string => typeof value === "string")
    : [];
}

function schemaGroup(context: MetricSeriesContext, group: MetricGroup): string[] {
  const groups = context.metricSchema?.groups;
  if (!groups || typeof groups !== "object" || Array.isArray(groups)) return [];
  const values = (groups as Record<string, unknown>)[group];
  return Array.isArray(values) ? values.filter((value): value is string => typeof value === "string") : [];
}

function taskSeriesKeys(points: MetricPoint[], context: MetricSeriesContext, task?: string): string[] {
  const primary = typeof context.metricSchema?.primary === "string" ? context.metricSchema.primary : undefined;
  const declared = uniqueKeys([primary, ...schemaMonitors(context)]);
  const defaults = task === "classification" ? [...CLASSIFICATION_SERIES] : task === "detection" ? [...DETECTION_SERIES] : [];
  if (!declared.length) return defaults.filter((key) => hasMetric(points, key));
  const keys = uniqueKeys(["loss", ...declared, ...defaults]).filter((key) => hasMetric(points, key));
  return hasMetric(points, "step_time_ms") && !keys.includes("step_time_ms") ? [...keys, "step_time_ms"] : keys;
}

export function deriveMetricSeries(points: MetricPoint[], context: MetricSeriesContext = {}): MetricSeriesSpec[] {
  const task = inferredTask(points, context);
  const keys = task === "classification" || task === "detection"
    ? taskSeriesKeys(points, context, task)
    : genericSeriesKeys(points, context);

  const declaredGroup = context.group ? schemaGroup(context, context.group) : [];
  const groupedKeys = declaredGroup.length
    ? declaredGroup.filter((key) => hasMetric(points, key))
    : context.group === "quality"
      ? keys.filter((key) => isRatioMetric(key))
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
      ? [primary, "accuracy"]
      : task === "detection"
        ? [primary, "mean_iou", "accuracy"]
        : [primary, "accuracy", "mean_iou"];

  const key = uniqueKeys(candidates).find((candidate) => candidate && hasMetric(points, candidate));
  if (!key) return undefined;
  return {
    key,
    label: signalLabel(key),
    format: metricFormat(key)
  };
}
