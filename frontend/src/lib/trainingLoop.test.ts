import { describe, expect, it } from "vitest";
import type { RunEvent } from "../api/types";
import { deriveTrainingLoopModel } from "./trainingLoop";

function stageEvent(
  stage: string,
  state: string,
  scope: "lifecycle" | "step" | "milestone",
  ts: number,
  step = 0
): RunEvent {
  return {
    event_id: `${scope}-${stage}-${state}-${ts}`,
    schema_version: "pulsegraph.event.v1",
    ts_ns: ts,
    source: "training",
    type: "training_stage",
    run_id: "run-1",
    step,
    payload: {
      scope,
      stage,
      state,
      message: `${stage} ${state}`,
      total_steps: 10,
      progress: step / 10
    }
  };
}

describe("deriveTrainingLoopModel", () => {
  it("uses runtime stage events as the source of truth", () => {
    const model = deriveTrainingLoopModel({
      hasResource: true,
      hasGraph: true,
      hasPrediction: false,
      metrics: [],
      events: [
        stageEvent("forward", "active", "step", 5, 3),
        stageEvent("data", "completed", "step", 4, 3),
        stageEvent("training", "active", "lifecycle", 3),
        stageEvent("loading", "completed", "lifecycle", 2),
        stageEvent("queued", "completed", "lifecycle", 1)
      ]
    });

    expect(model.eventDriven).toBe(true);
    expect(model.activeStage).toMatchObject({ id: "forward", scope: "step", state: "active", evidence: "ops" });
    expect(model.currentStep).toBe(3);
    expect(model.totalSteps).toBe(10);
    expect(model.stepLoop.find((stage) => stage.id === "data")?.state).toBe("completed");
  });

  it("exposes opaque resource-owned steps without inventing internal stages", () => {
    const model = deriveTrainingLoopModel({
      hasResource: true,
      hasGraph: true,
      hasPrediction: false,
      metrics: [],
      events: [stageEvent("custom_step", "active", "step", 2, 1)]
    });

    expect(model.stepLoop[model.stepLoop.length - 1]).toMatchObject({ id: "custom_step", label: "Custom Step", state: "active" });
  });

  it("falls back to legacy evidence for recorded runs without stage events", () => {
    const model = deriveTrainingLoopModel({
      hasResource: true,
      hasGraph: true,
      hasPrediction: false,
      metrics: [{ step: 2, loss: 0.4, learningRate: 0.01, values: { loss: 0.4 } }],
      events: []
    });

    expect(model.eventDriven).toBe(false);
    expect(model.stepLoop.find((stage) => stage.id === "loss")).toMatchObject({ state: "completed", detail: "Loss 0.4000" });
    expect(model.stepLoop.find((stage) => stage.id === "optimizer")?.state).toBe("completed");
  });

  it("does not describe graph readiness as a completed training step before a run", () => {
    const model = deriveTrainingLoopModel({
      hasResource: true,
      hasGraph: true,
      hasPrediction: false,
      metrics: [],
      events: []
    });

    expect(model.activeStage).toBeUndefined();
    expect(model.message).toBe("Resource ready to train");
  });
});
