import { describe, expect, it } from "vitest";
import { applyStreamEvent, createInitialStreamState } from "./useRunStream";
import type { RunEvent } from "../api/client";

function metricEvent(step: number, payload: RunEvent["payload"]): RunEvent {
  return {
    event_id: `metric-${step}-${Object.keys(payload).join("-")}`,
    schema_version: "1",
    ts_ns: 1,
    source: "training",
    run_id: "run-1",
    step,
    type: "metric",
    layer: null,
    payload
  } as RunEvent;
}

function infraEvent(step: number, payload: RunEvent["payload"]): RunEvent {
  return {
    event_id: `infra-${step}-${Object.keys(payload).join("-")}`,
    schema_version: "1",
    ts_ns: 1,
    source: "infra",
    run_id: "run-1",
    step,
    type: "infra",
    layer: null,
    payload
  } as RunEvent;
}

describe("applyStreamEvent", () => {
  it("captures arbitrary numeric metric keys while preserving classification shortcuts", () => {
    const next = applyStreamEvent(
      createInitialStreamState(),
      metricEvent(7, { loss: 1.4, accuracy: 0.76, learning_rate: 0.001, mean_iou: 0.52, loss_box_reg: 0.18 })
    );

    expect(next.metrics).toHaveLength(1);
    expect(next.metrics[0]).toMatchObject({
      step: 7,
      loss: 1.4,
      accuracy: 0.76,
      learningRate: 0.001
    });
    expect(next.metrics[0].values).toEqual({
      loss: 1.4,
      accuracy: 0.76,
      learning_rate: 0.001,
      mean_iou: 0.52,
      loss_box_reg: 0.18
    });
  });

  it("merges infra telemetry into the same step without dropping generic metric values", () => {
    const withMetrics = applyStreamEvent(createInitialStreamState(), metricEvent(3, { loss: 0.9, loss_classifier: 0.33 }));
    const next = applyStreamEvent(
      withMetrics,
      infraEvent(3, { step_time_ms: 18, memory_peak_mb: 512, samples_per_sec: 144, device: "cpu" })
    );

    expect(next.device).toBe("cpu");
    expect(next.metrics[0]).toMatchObject({
      step: 3,
      loss: 0.9,
      stepTimeMs: 18,
      memoryPeakMb: 512
    });
    expect(next.metrics[0].values).toEqual({
      loss: 0.9,
      loss_classifier: 0.33,
      step_time_ms: 18,
      memory_peak_mb: 512,
      samples_per_sec: 144
    });
  });
});
