import { describe, expect, it } from "vitest";
import type { ModelGraph, RunEvent } from "../api/client";
import {
  deriveCausalFocus,
  deriveTimelineFrames,
  eventsAtTimelineStep,
  layerSnapshotsAtStep,
  peakLossStep,
  resolveTimelineStep
} from "./timeline";

const baseEvent = {
  schema_version: "1",
  ts_ns: 1,
  source: "training",
  run_id: "run-1",
  epoch: null
} as const;

function event(step: number, type: RunEvent["type"], layer?: string): RunEvent {
  if (type === "layer_snapshot") {
    return {
      ...baseEvent,
      event_id: `${type}-${step}-${layer ?? "none"}`,
      type,
      step,
      layer,
      payload: { gradient_norm: 0.2, activation_sparsity: 0.1 }
    };
  }
  if (type === "checkpoint") {
    return { ...baseEvent, event_id: `${type}-${step}`, type, step, layer: null, payload: { path: "ckpt.pt" } };
  }
  return { ...baseEvent, event_id: `${type}-${step}`, type: "metric", step, layer: null, payload: { loss: 1 } };
}

describe("timeline helpers", () => {
  it("builds sparse scrubber frames from metrics, events, and layer history", () => {
    const frames = deriveTimelineFrames(
      [{ step: 5, loss: 0.7, learningRate: 0.01 }],
      [event(10, "checkpoint")],
      { conv1: [{ step: 15, gradient_norm: 0.1 }] }
    );

    expect(frames.map((frame) => frame.step)).toEqual([5, 10, 15]);
    expect(frames[0].metric?.learningRate).toBe(0.01);
    expect(resolveTimelineStep(frames, 11)).toBe(10);
    expect(resolveTimelineStep(frames)).toBe(15);
  });

  it("reconstructs layer snapshots at the selected step", () => {
    const snapshots = layerSnapshotsAtStep(
      { conv1: { layer_id: "conv1", gradient_norm: 9, activation_sparsity: 0.2 } },
      { conv1: [{ step: 1, gradient_norm: 0.1 }, { step: 4, gradient_norm: 0.5 }] },
      3
    );

    expect(snapshots.conv1.gradient_norm).toBe(0.1);
    expect(snapshots.conv1.activation_sparsity).toBe(0.2);
  });

  it("filters event details to the selected frame", () => {
    const events = [event(10, "checkpoint"), event(5, "layer_snapshot", "conv1")];
    expect(eventsAtTimelineStep(events, 5).map((item) => item.event_id)).toEqual(["layer_snapshot-5-conv1"]);
    expect(eventsAtTimelineStep(events).length).toBe(2);
  });

  it("surfaces causal focus for unhealthy layers and loss peaks", () => {
    const graph: ModelGraph = {
      nodes: [
        { id: "conv1", label: "Conv 1", kind: "Conv2d", param_count: 16, confidence: "trusted", metadata: {} }
      ],
      edges: []
    };
    const focus = deriveCausalFocus({
      step: 20,
      metrics: [{ step: 10, loss: 1.2 }, { step: 20, loss: 3.4 }],
      events: [],
      graph,
      layerSnapshots: { conv1: { layer_id: "conv1", gradient_norm: 0.00000001, activation_sparsity: 0.1 } }
    });

    expect(focus.layerId).toBe("conv1");
    expect(focus.severity).toBe("warning");
    expect(peakLossStep([{ step: 1, loss: 0.4 }, { step: 2, loss: 0.9 }])).toBe(2);
  });
});
