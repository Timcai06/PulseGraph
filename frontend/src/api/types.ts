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
  weights_fingerprint?: string | null;
  matched_run_id?: string | null;
  filename: string;
  mode: "state_dict" | "checkpoint" | "safetensors" | "torchscript" | "unknown";
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
  sample_source: "mnist" | "synthetic" | "probe";
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

export type GraphPayload = {
  graph?: ModelGraph | null;
  tensors?: { name: string; shape: number[] }[] | null;
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
  | (RunEventBase & { type: "graph"; payload: GraphPayload })
  | (RunEventBase & { type: "source_registered"; payload: Record<string, unknown> })
  | (RunEventBase & { type: "config_registered"; payload: Record<string, unknown> })
  | (RunEventBase & { type: "graph_registered"; payload: Record<string, unknown> })
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

export type CheckpointInfo = {
  step: number;
  epoch?: number | null;
  path: string;
  size_mb: number;
  fingerprint?: string | null;
};

export type RunDetail = {
  run_id: string;
  created_at: number;
  completed: boolean;
  source?: string | null;
  entry_class?: string | null;
  source_files: string[];
  source_origin?: "recorded" | "user-attached" | null;
  config?: Record<string, unknown> | null;
  graph?: ModelGraph | null;
  has_samples: boolean;
  metrics: Record<string, unknown>[];
  checkpoints: CheckpointInfo[];
  event_count: number;
};

export type RunInsight = {
  severity: "info" | "warning" | "critical";
  title: string;
  detail: string;
  suggestion?: string | null;
};

export type LayerHealth = {
  layer_id: string;
  mean_sparsity?: number | null;
  last_gradient_norm?: number | null;
  gradient_trend: "stable" | "vanishing" | "exploding" | "unknown";
  weight_std_drift?: number | null;
};

export type CheckpointEvaluation = {
  step: number;
  accuracy?: number | null;
  sample_count: number;
};

export type ErrorAnalysis = {
  confusion: number[][];
  labels: number[];
  misclassified: { index: number; label: number; prediction: number; pixels?: number[] }[];
};

export type SourceCandidate = {
  class_name: string;
  file: string;
};

export type SourceValidationResult = {
  ok: boolean;
  error?: string | null;
  missing_keys: string[];
  unexpected_keys: string[];
};

export type AttachSourceResult = {
  run_id: string;
  saved: string[];
  entry_file: string;
  entry_class: string;
  validation?: SourceValidationResult | null;
};

export type ImportArtifactResult = {
  run_id: string;
  fingerprint: string;
  created: boolean;
};

export type SourceImportResult = {
  run_id: string;
  run_kind: "source-import" | "source-training";
  inference_only: boolean;
  saved: string[];
  entry_file: string;
  entry_class: string;
  graph: ModelGraph;
  checkpoint: {
    step: number;
    path: string;
    fingerprint: string;
  };
};

export type RunReport = {
  run_id: string;
  generated_for_checkpoint?: number | null;
  final_loss?: number | null;
  best_accuracy?: number | null;
  overfit_gap?: number | null;
  loss_plateau_step?: number | null;
  layer_health: LayerHealth[];
  checkpoint_evaluations: CheckpointEvaluation[];
  error_analysis?: ErrorAnalysis | null;
  insights: RunInsight[];
};
