import type { RunEvent } from "../api/client";
import type { MetricPoint } from "../hooks/useRunStream";

export type TrainingStageScope = "lifecycle" | "step" | "milestone";
export type TrainingStageState = "pending" | "active" | "completed" | "warning" | "failed" | "cancelled";
export type TrainingStageEvidence = "prepare" | "ops" | "telemetry" | "diagnostics" | "checkpoint" | "evaluate";

export type TrainingLoopStage = {
  id: string;
  label: string;
  scope: TrainingStageScope;
  state: TrainingStageState;
  detail: string;
  step?: number;
  durationMs?: number;
  evidence?: TrainingStageEvidence;
};

export type TrainingLoopModel = {
  lifecycle: TrainingLoopStage[];
  stepLoop: TrainingLoopStage[];
  milestones: TrainingLoopStage[];
  activeStage?: TrainingLoopStage;
  currentStep: number;
  totalSteps?: number;
  progress?: number;
  message: string;
  eventDriven: boolean;
};

type Input = {
  hasResource: boolean;
  hasGraph: boolean;
  hasPrediction: boolean;
  metrics: MetricPoint[];
  events: RunEvent[];
  trainingStageEvents?: Array<Extract<RunEvent, { type: "training_stage" }>>;
  progress?: {
    step?: number;
    totalSteps?: number;
    progress?: number;
    phase?: string;
    message?: string;
  };
};

type StageDefinition = Pick<TrainingLoopStage, "id" | "label" | "scope" | "evidence">;

const lifecycleDefinitions: StageDefinition[] = [
  { id: "queued", label: "Queued", scope: "lifecycle" },
  { id: "loading", label: "Load", scope: "lifecycle", evidence: "prepare" },
  { id: "building", label: "Build", scope: "lifecycle", evidence: "ops" },
  { id: "preparing_data", label: "Prepare", scope: "lifecycle", evidence: "prepare" },
  { id: "initializing", label: "Initialize", scope: "lifecycle", evidence: "telemetry" },
  { id: "training", label: "Train", scope: "lifecycle", evidence: "telemetry" },
  { id: "checkpointing", label: "Finalize", scope: "lifecycle", evidence: "checkpoint" },
  { id: "completed", label: "Complete", scope: "lifecycle", evidence: "evaluate" }
];

const stepDefinitions: StageDefinition[] = [
  { id: "data", label: "Batch", scope: "step", evidence: "prepare" },
  { id: "forward", label: "Forward", scope: "step", evidence: "ops" },
  { id: "loss", label: "Loss", scope: "step", evidence: "telemetry" },
  { id: "backward", label: "Backward", scope: "step", evidence: "telemetry" },
  { id: "optimizer", label: "Update", scope: "step", evidence: "telemetry" }
];

const milestoneDefinitions: StageDefinition[] = [
  { id: "checkpoint", label: "Checkpoint", scope: "milestone", evidence: "checkpoint" },
  { id: "evaluation", label: "Evaluation", scope: "milestone", evidence: "evaluate" }
];

function stageState(value: unknown): TrainingStageState {
  return ["active", "completed", "warning", "failed", "cancelled"].includes(String(value))
    ? String(value) as TrainingStageState
    : "pending";
}

function stageKey(scope: string, stage: string) {
  return `${scope}:${stage}`;
}

function fromEvent(definition: StageDefinition, event?: Extract<RunEvent, { type: "training_stage" }>): TrainingLoopStage {
  return {
    ...definition,
    state: stageState(event?.payload.state),
    detail: event?.payload.message ?? "Waiting",
    step: event?.step,
    durationMs: typeof event?.payload.duration_ms === "number" ? event.payload.duration_ms : undefined
  };
}

function latestStageEvents(events: RunEvent[]) {
  const stageEvents = events
    .filter((event): event is Extract<RunEvent, { type: "training_stage" }> => event.type === "training_stage");
  const latest = new Map<string, Extract<RunEvent, { type: "training_stage" }>>();
  for (const event of stageEvents) {
    const key = stageKey(event.payload.scope, event.payload.stage);
    if (!latest.has(key)) latest.set(key, event);
  }
  return { stageEvents, latest };
}

function deriveEventDrivenModel(input: Input): TrainingLoopModel | undefined {
  const { stageEvents, latest } = latestStageEvents(input.trainingStageEvents ?? input.events);
  if (!stageEvents.length) return undefined;

  const customStep = latest.get(stageKey("step", "custom_step"));
  const stepStageDefinitions = customStep
    ? [...stepDefinitions, { id: "custom_step", label: "Custom Step", scope: "step" as const, evidence: "telemetry" as const }]
    : stepDefinitions;
  const lifecycle = lifecycleDefinitions.map((definition) => fromEvent(definition, latest.get(stageKey(definition.scope, definition.id))));
  const stepLoop = stepStageDefinitions.map((definition) => fromEvent(definition, latest.get(stageKey(definition.scope, definition.id))));
  const milestones = milestoneDefinitions.map((definition) => fromEvent(definition, latest.get(stageKey(definition.scope, definition.id))));
  const allStages = [...lifecycle, ...stepLoop, ...milestones];
  const latestActiveEvent = stageEvents.find((event) => {
    const current = latest.get(stageKey(event.payload.scope, event.payload.stage));
    return current?.event_id === event.event_id && event.payload.state === "active";
  });
  const newest = stageEvents[0];
  const newestStage = allStages.find((stage) => stage.scope === newest.payload.scope && stage.id === newest.payload.stage);
  const activeStage = latestActiveEvent
    ? allStages.find((stage) => stage.scope === latestActiveEvent.payload.scope && stage.id === latestActiveEvent.payload.stage)
    : allStages.find((stage) => stage.state === "failed" || stage.state === "cancelled")
      ?? newestStage
      ?? [...allStages].reverse().find((stage) => stage.state === "completed");
  const totalSteps = newest.payload.total_steps ?? input.progress?.totalSteps;
  const currentStep = Math.max(newest.step, input.progress?.step ?? 0, input.metrics[input.metrics.length - 1]?.step ?? 0);

  return {
    lifecycle,
    stepLoop,
    milestones,
    activeStage,
    currentStep,
    totalSteps: typeof totalSteps === "number" ? totalSteps : undefined,
    progress: typeof newest.payload.progress === "number" ? newest.payload.progress : input.progress?.progress,
    message: activeStage?.detail ?? newest.payload.message,
    eventDriven: true
  };
}

function legacyStage(definition: StageDefinition, state: TrainingStageState, detail: string): TrainingLoopStage {
  return { ...definition, state, detail };
}

function deriveLegacyModel(input: Input): TrainingLoopModel {
  const latestMetric = input.metrics[input.metrics.length - 1];
  const hasLayerSnapshot = input.events.some((event) => event.type === "layer_snapshot");
  const hasCheckpoint = input.events.some((event) => event.type === "checkpoint");
  const hasRunComplete = input.events.some((event) => event.type === "run_complete");
  const hasTrainingEvidence = Boolean(latestMetric || hasLayerSnapshot || hasCheckpoint || hasRunComplete);
  const phase = input.progress?.phase;
  const lifecycle = lifecycleDefinitions.map((definition) => {
    if (definition.id === "loading") return legacyStage(definition, input.hasResource ? "completed" : "pending", input.hasResource ? "Resource loaded" : "Waiting for resource");
    if (definition.id === "building") return legacyStage(definition, input.hasResource && input.hasGraph ? "completed" : "pending", input.hasResource && input.hasGraph ? "Operator graph ready" : "Waiting for resource graph");
    if (definition.id === "training") return legacyStage(definition, hasRunComplete ? "completed" : latestMetric ? "active" : "pending", latestMetric ? `Step ${latestMetric.step}` : "Waiting for metrics");
    if (definition.id === "completed") return legacyStage(definition, hasRunComplete ? "completed" : "pending", hasRunComplete ? "Run complete" : "Waiting");
    return legacyStage(definition, phase === definition.id ? "active" : "pending", phase === definition.id ? input.progress?.message ?? definition.label : "Waiting");
  });
  const stepLoop = stepDefinitions.map((definition) => {
    if (definition.id === "data") return legacyStage(definition, input.hasResource ? "completed" : "pending", input.hasResource ? "Batch source ready" : "Waiting");
    if (definition.id === "forward") return legacyStage(definition, input.hasResource && input.hasGraph ? "completed" : "pending", input.hasResource && input.hasGraph ? "Graph evidence available" : "Waiting");
    if (definition.id === "loss") return legacyStage(definition, latestMetric?.loss == null ? "pending" : "completed", latestMetric?.loss == null ? "Waiting" : `Loss ${latestMetric.loss.toFixed(4)}`);
    if (definition.id === "backward") return legacyStage(definition, hasLayerSnapshot ? "completed" : "pending", hasLayerSnapshot ? "Layer evidence available" : "Waiting");
    return legacyStage(definition, latestMetric?.learningRate == null ? "pending" : "completed", latestMetric?.learningRate == null ? "Waiting" : `LR ${latestMetric.learningRate}`);
  });
  const milestones = [
    legacyStage(milestoneDefinitions[0], hasCheckpoint ? "completed" : "pending", hasCheckpoint ? "Checkpoint recorded" : "Waiting"),
    legacyStage(milestoneDefinitions[1], input.hasPrediction || hasRunComplete ? "completed" : "pending", input.hasPrediction ? "Prediction ready" : hasRunComplete ? "Run complete" : "Waiting")
  ];
  const allStages = [...lifecycle, ...stepLoop, ...milestones];
  const activeStage = allStages.find((stage) => stage.state === "active")
    ?? (hasTrainingEvidence ? [...allStages].reverse().find((stage) => stage.state === "completed") : undefined);

  return {
    lifecycle,
    stepLoop,
    milestones,
    activeStage,
    currentStep: input.progress?.step ?? latestMetric?.step ?? 0,
    totalSteps: input.progress?.totalSteps,
    progress: input.progress?.progress,
    message: input.progress?.message
      ?? activeStage?.detail
      ?? (input.hasResource && input.hasGraph ? "Resource ready to train" : "Waiting for training evidence"),
    eventDriven: false
  };
}

export function deriveTrainingLoopModel(input: Input): TrainingLoopModel {
  return deriveEventDrivenModel(input) ?? deriveLegacyModel(input);
}
