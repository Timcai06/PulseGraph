import type { RunEvent } from "../api/client";
import type { MetricPoint } from "../hooks/useRunStream";

export type TrainingLoopStage = {
  id: "data" | "forward" | "loss" | "backward" | "optimizer" | "checkpoint" | "eval";
  label: string;
  state: "idle" | "active" | "healthy" | "warning";
  detail: string;
};

type Input = {
  hasResource: boolean;
  hasGraph: boolean;
  hasPrediction: boolean;
  metrics: MetricPoint[];
  events: RunEvent[];
  learningRate?: number | null;
};

export function deriveTrainingLoopStages(input: Input): TrainingLoopStage[] {
  const latestMetric = input.metrics.length ? input.metrics[input.metrics.length - 1] : undefined;
  const hasLayerSnapshot = input.events.some((event) => event.type === "layer_snapshot");
  const hasCheckpoint = input.events.some((event) => event.type === "checkpoint");
  const hasRunComplete = input.events.some((event) => event.type === "run_complete");

  return [
    {
      id: "data",
      label: "Data",
      state: input.hasResource ? "healthy" : "idle",
      detail: input.hasResource ? "resource loaded" : "waiting for resource"
    },
    {
      id: "forward",
      label: "Forward",
      state: input.hasGraph ? "active" : "idle",
      detail: input.hasGraph ? "operator graph ready" : "no graph"
    },
    {
      id: "loss",
      label: "Loss",
      state: latestMetric?.loss == null ? "idle" : "active",
      detail: latestMetric?.loss == null ? "no loss yet" : `loss ${latestMetric.loss.toFixed(4)}`
    },
    {
      id: "backward",
      label: "Backward",
      state: hasLayerSnapshot ? "active" : "idle",
      detail: hasLayerSnapshot ? "layer snapshots flowing" : "no gradient evidence"
    },
    {
      id: "optimizer",
      label: "Optimizer",
      state: input.learningRate == null ? "idle" : "healthy",
      detail: input.learningRate == null ? "learning rate unknown" : `lr ${input.learningRate}`
    },
    {
      id: "checkpoint",
      label: "Checkpoint",
      state: hasCheckpoint ? "healthy" : "idle",
      detail: hasCheckpoint ? "checkpoint recorded" : "no checkpoint"
    },
    {
      id: "eval",
      label: "Eval",
      state: input.hasPrediction || hasRunComplete ? "healthy" : "idle",
      detail: input.hasPrediction ? "prediction ready" : hasRunComplete ? "run complete" : "waiting"
    }
  ];
}
