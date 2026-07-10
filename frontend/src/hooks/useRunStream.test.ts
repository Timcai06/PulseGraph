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

function trainingStageEvent(
  eventId: string,
  step: number,
  stage: string,
  state: "active" | "completed"
): Extract<RunEvent, { type: "training_stage" }> {
  return {
    event_id: eventId,
    schema_version: "1",
    ts_ns: 1,
    source: "training",
    run_id: "run-1",
    step,
    type: "training_stage",
    layer: null,
    payload: { scope: "lifecycle", stage, state, message: `${stage} ${state}` }
  };
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

  it("tracks lifecycle progress separately from metric points", () => {
    const next = applyStreamEvent(createInitialStreamState(), {
      event_id: "status-12",
      schema_version: "1",
      ts_ns: 1,
      source: "training",
      run_id: "run-1",
      step: 12,
      type: "run_status",
      layer: null,
      payload: {
        phase: "training",
        message: "Training step 12 of 100",
        step: 12,
        total_steps: 100,
        progress: 0.12,
        elapsed_sec: 4,
        eta_sec: 29
      }
    });

    expect(next.progress).toEqual({
      phase: "training",
      message: "Training step 12 of 100",
      step: 12,
      totalSteps: 100,
      progress: 0.12,
      elapsedSec: 4,
      etaSec: 29
    });
    expect(next.metrics).toEqual([]);
  });

  it("preserves the latest state for every training stage outside the bounded event feed", () => {
    let next = applyStreamEvent(createInitialStreamState(), trainingStageEvent("load-active", 0, "loading", "active"));
    next = applyStreamEvent(next, trainingStageEvent("load-complete", 0, "loading", "completed"));
    next = applyStreamEvent(next, trainingStageEvent("train-active", 1, "training", "active"));

    for (let step = 0; step < 70; step += 1) {
      next = applyStreamEvent(next, metricEvent(step, { loss: 1 / (step + 1) }));
    }

    expect(next.events).toHaveLength(60);
    expect(next.trainingStages).toHaveLength(2);
    expect(next.trainingStages.map((event) => `${event.payload.stage}:${event.payload.state}`)).toEqual([
      "training:active",
      "loading:completed"
    ]);
  });

  it("expands one aggregate layer event into sampled node snapshots", () => {
    const next = applyStreamEvent(createInitialStreamState(), {
      event_id: "layers-5",
      schema_version: "1",
      ts_ns: 1,
      source: "runtime_hook",
      run_id: "run-1",
      step: 5,
      type: "layer_snapshot",
      layer: "__aggregate__",
      payload: {
        mode: "aggregate",
        layer_count: 125,
        layers: [
          { layer_id: "backbone.0", activation_mean: 0.2, gradient_norm: 1.4 },
          { layer_id: "head", activation_sparsity: 0.6, gradient_norm: 0.8 }
        ]
      }
    });

    expect(Object.keys(next.layerSnapshots)).toEqual(["backbone.0", "head"]);
    expect(next.layerHistory["backbone.0"]).toEqual([{ step: 5, activation_mean: 0.2, gradient_norm: 1.4 }]);
    expect(next.pulsedNodeId).toBe("backbone.0");
  });

  it("surfaces failed completion as an error state", () => {
    const next = applyStreamEvent(createInitialStreamState(), {
      event_id: "failed",
      schema_version: "1",
      ts_ns: 1,
      source: "training",
      run_id: "run-1",
      step: 8,
      type: "run_complete",
      layer: null,
      payload: { status: "failed", error: "out of memory" }
    });

    expect(next.status).toBe("error");
    expect(next.progress).toMatchObject({ phase: "failed", step: 8 });
  });
});
