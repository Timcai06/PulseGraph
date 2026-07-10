import { describe, expect, it } from "vitest";
import { deriveMetricSeries, derivePrimaryMetricSignal, latestMetricValue } from "./metricSeries";
import type { MetricPoint } from "../hooks/useRunStream";

function point(step: number, values: Record<string, number>, extra: Partial<MetricPoint> = {}): MetricPoint {
  return {
    step,
    values,
    loss: values.loss,
    accuracy: values.accuracy,
    learningRate: values.learning_rate,
    stepTimeMs: values.step_time_ms,
    memoryPeakMb: values.memory_peak_mb,
    ...extra
  };
}

describe("metricSeries", () => {
  it("keeps classification telemetry focused on loss, accuracy, and step time", () => {
    const points = [
      point(1, { loss: 1.2, accuracy: 0.44, step_time_ms: 12, learning_rate: 0.001 }),
      point(2, { loss: 0.9, accuracy: 0.57, step_time_ms: 11, learning_rate: 0.001 })
    ];

    expect(deriveMetricSeries(points, { task: "classification" }).map((series) => series.key)).toEqual([
      "loss",
      "accuracy",
      "step_time_ms"
    ]);
  });

  it("surfaces bounded detection metrics with mean IoU and total loss", () => {
    const points = [
      point(5, {
        loss: 1.8,
        loss_classifier: 0.6,
        loss_box_reg: 0.3,
        mean_iou: 0.41,
        step_time_ms: 25,
        samples_per_sec: 90
      })
    ];

    expect(deriveMetricSeries(points, { task: "detection" }).map((series) => series.key)).toEqual([
      "loss",
      "loss_classifier",
      "loss_box_reg",
      "mean_iou",
      "step_time_ms"
    ]);
    expect(derivePrimaryMetricSignal(points, { task: "detection" })).toEqual({
      key: "mean_iou",
      label: "mean IoU",
      format: "percent"
    });
  });

  it("uses metric schema hints for non-classification primary signals", () => {
    const points = [point(3, { loss: 0.5, f1_score: 0.82, step_time_ms: 8 })];

    expect(deriveMetricSeries(points, { metricSchema: { primary: "f1_score", monitors: ["loss", "f1_score"] } }).map((series) => series.key)).toEqual([
      "loss",
      "f1_score",
      "step_time_ms"
    ]);
    expect(derivePrimaryMetricSignal(points, { metricSchema: { primary: "f1_score" } })).toEqual({
      key: "f1_score",
      label: "F1 Score",
      format: "float"
    });
    expect(latestMetricValue(points, "f1_score")).toBe(0.82);
  });

  it("separates optimization, quality, and infra metric groups", () => {
    const points = [point(5, {
      loss: 1.8,
      loss_classifier: 0.6,
      loss_box_reg: 0.3,
      mean_iou: 0.41,
      step_time_ms: 25,
      samples_per_sec: 90,
      memory_peak_mb: 512
    })];

    expect(deriveMetricSeries(points, { task: "detection", group: "optimization" }).map((series) => series.key)).toEqual([
      "loss",
      "loss_classifier",
      "loss_box_reg"
    ]);
    expect(deriveMetricSeries(points, { task: "detection", group: "quality" }).map((series) => series.key)).toEqual(["mean_iou"]);
    expect(deriveMetricSeries(points, { task: "detection", group: "infra" }).map((series) => series.key)).toEqual([
      "step_time_ms",
      "samples_per_sec",
      "memory_peak_mb"
    ]);
  });
});
