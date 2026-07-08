export type GraphNode = {
  id: string;
  label: string;
  kind: string;
  input_shape?: number[] | null;
  output_shape?: number[] | null;
  param_count: number;
  confidence: "safe" | "inferred" | "trusted";
  metadata: Record<string, unknown>;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  label?: string | null;
};

export type ModelGraph = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type TensorSummary = {
  name: string;
  shape: number[];
  dtype: string;
  numel: number;
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
};

export type InspectionResponse = {
  artifact_id?: string | null;
  artifact_sha256?: string | null;
  filename: string;
  mode: "state_dict" | "checkpoint" | "unknown";
  safe: boolean;
  tensors: TensorSummary[];
  graph: ModelGraph;
  warnings: string[];
};

export type LayerSnapshot = {
  layer_id: string;
  input_shape?: number[] | null;
  output_shape?: number[] | null;
  activation_mean?: number | null;
  activation_sparsity?: number | null;
  gradient_norm?: number | null;
  weight_std?: number | null;
};

export type PredictionResponse = {
  sample_index: number;
  label: number;
  prediction: number;
  weights: "trained" | "random";
  sample_source: "mnist" | "synthetic";
  image_pixels: number[];
  probabilities: number[];
  graph: ModelGraph;
  layers: LayerSnapshot[];
};

export type MetricPayload = {
  loss?: number | null;
  accuracy?: number | null;
  learning_rate?: number | null;
};

export type LayerSnapshotPayload = {
  activation_mean?: number | null;
  activation_sparsity?: number | null;
  gradient_norm?: number | null;
  weight_std?: number | null;
};

export type InfraPayload = {
  device?: string | null;
  step_time_ms?: number | null;
  samples_per_sec?: number | null;
  memory_peak_mb?: number | null;
  elapsed_sec?: number | null;
};

export type CheckpointPayload = {
  path?: string | null;
  size_mb?: number | null;
  save_time_ms?: number | null;
};

export type AnimationPayload = {
  name?: string | null;
  path?: string[] | null;
};

type RunEventBase = {
  event_id: string;
  schema_version: string;
  ts_ns: number;
  source: "training" | "runtime_hook" | "checkpoint" | "infra" | "plugin" | "animation";
  run_id: string;
  session_id?: string | null;
  step: number;
  epoch?: number | null;
  layer?: string | null;
};

export type RunEvent =
  | (RunEventBase & { type: "metric"; payload: MetricPayload })
  | (RunEventBase & { type: "layer_snapshot"; payload: LayerSnapshotPayload })
  | (RunEventBase & { type: "infra"; payload: InfraPayload })
  | (RunEventBase & { type: "checkpoint"; payload: CheckpointPayload })
  | (RunEventBase & { type: "animation"; payload: AnimationPayload })
  | (RunEventBase & { type: "run_complete"; payload: Record<string, unknown> });

export type RunEventType = RunEvent["type"];

export type RunSummary = {
  run_id: string;
  created_at: number;
  last_event_at: number;
  completed: boolean;
  event_count: number;
  last_step: number;
};
