import type { LayerSnapshot, ModelGraph, RunEvent } from "../api/client";
import type { MetricPoint } from "../hooks/useRunStream";
import { deriveLayerHealth } from "./layerHealth";

type LayerHistoryInput = Record<string, Array<{ step: number } & Partial<LayerSnapshot>>>;

export type TimelineFrame = {
  step: number;
  metric?: MetricPoint;
  eventCount: number;
  layerCount: number;
};

export type CausalFocus = {
  step: number;
  severity: "info" | "warning" | "critical";
  title: string;
  detail: string;
  layerId?: string;
  eventId?: string;
  jumpStep?: number;
};

export function deriveTimelineFrames(
  metrics: MetricPoint[],
  events: RunEvent[],
  layerHistory: LayerHistoryInput
): TimelineFrame[] {
  const steps = new Set<number>();
  for (const point of metrics) steps.add(point.step);
  for (const event of events) steps.add(event.step);
  for (const history of Object.values(layerHistory)) {
    for (const point of history) steps.add(point.step);
  }

  return Array.from(steps)
    .sort((a, b) => a - b)
    .map((step) => ({
      step,
      metric: metrics.find((point) => point.step === step),
      eventCount: events.filter((event) => event.step === step).length,
      layerCount: Object.values(layerHistory).filter((history) => history.some((point) => point.step === step)).length
    }));
}

export function resolveTimelineStep(frames: TimelineFrame[], selectedStep?: number): number | undefined {
  if (!frames.length) return undefined;
  if (selectedStep == null) return frames[frames.length - 1].step;
  return frames.reduce((closest, frame) =>
    Math.abs(frame.step - selectedStep) < Math.abs(closest.step - selectedStep) ? frame : closest
  ).step;
}

export function layerSnapshotsAtStep(
  latestSnapshots: Record<string, LayerSnapshot>,
  layerHistory: LayerHistoryInput,
  step?: number
): Record<string, LayerSnapshot> {
  if (step == null) return latestSnapshots;
  const next: Record<string, LayerSnapshot> = {};
  for (const [layerId, latest] of Object.entries(latestSnapshots)) {
    const history = layerHistory[layerId] ?? [];
    const historical = [...history].reverse().find((point) => point.step <= step);
    next[layerId] = historical ? { ...latest, ...historical, layer_id: layerId } : latest;
  }
  for (const [layerId, history] of Object.entries(layerHistory)) {
    if (next[layerId]) continue;
    const historical = [...history].reverse().find((point) => point.step <= step);
    if (historical) next[layerId] = { ...historical, layer_id: layerId };
  }
  return next;
}

export function eventsAtTimelineStep(events: RunEvent[], step?: number): RunEvent[] {
  if (step == null) return events;
  const exact = events.filter((event) => event.step === step);
  return exact.length ? exact : events.filter((event) => event.step <= step).slice(0, 12);
}

function closestMetric(metrics: MetricPoint[], step?: number): MetricPoint | undefined {
  if (!metrics.length) return undefined;
  if (step == null) return metrics[metrics.length - 1];
  return [...metrics].reverse().find((point) => point.step <= step) ?? metrics[0];
}

export function peakLossStep(metrics: MetricPoint[]): number | undefined {
  const losses = metrics.filter((point) => point.loss != null);
  if (!losses.length) return undefined;
  return losses.reduce((peak, point) => (Number(point.loss) > Number(peak.loss) ? point : peak)).step;
}

export function deriveCausalFocus(input: {
  step?: number;
  metrics: MetricPoint[];
  events: RunEvent[];
  graph: ModelGraph;
  layerSnapshots: Record<string, LayerSnapshot>;
}): CausalFocus {
  const step = input.step ?? 0;
  const metric = closestMetric(input.metrics, input.step);
  const peakStep = peakLossStep(input.metrics);
  const eventAtStep = input.events.find((event) => event.step === step && event.type !== "metric");

  let worstLayer: { id: string; severity: CausalFocus["severity"]; label: string; detail: string } | undefined;
  for (const node of input.graph.nodes) {
    const snapshot = input.layerSnapshots[node.id];
    const health = deriveLayerHealth(node, snapshot);
    const severity: CausalFocus["severity"] = health.severity === "warning" || health.severity === "caution" ? "warning" : "info";
    if (severity === "info") continue;
    if (!worstLayer) {
      worstLayer = { id: node.id, severity, label: node.label || node.id, detail: health.detail };
    }
  }

  if (worstLayer) {
    return {
      step,
      severity: worstLayer.severity,
      title: `Layer focus: ${worstLayer.label}`,
      detail: worstLayer.detail,
      layerId: worstLayer.id,
      jumpStep: peakStep
    };
  }

  if (metric?.loss != null && peakStep === metric.step && input.metrics.length > 1) {
    return {
      step,
      severity: "warning",
      title: "Loss peak",
      detail: `Selected frame is the highest observed loss (${metric.loss.toFixed(4)}).`,
      eventId: eventAtStep?.event_id,
      jumpStep: peakStep
    };
  }

  if (eventAtStep) {
    return {
      step,
      severity: eventAtStep.type === "checkpoint" || eventAtStep.type === "run_complete" ? "info" : "warning",
      title: eventAtStep.type.replace(/_/g, " "),
      detail: eventAtStep.layer ? `${eventAtStep.layer} emitted at step ${eventAtStep.step}.` : `Runtime event at step ${eventAtStep.step}.`,
      layerId: eventAtStep.layer ?? undefined,
      eventId: eventAtStep.event_id,
      jumpStep: peakStep
    };
  }

  return {
    step,
    severity: "info",
    title: metric?.loss == null ? "Waiting for signal" : "Training frame stable",
    detail: metric?.loss == null ? "No loss or layer anomaly is available for this frame." : `Loss ${metric.loss.toFixed(4)} at step ${metric.step}.`,
    jumpStep: peakStep
  };
}
