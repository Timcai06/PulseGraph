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
  image_pixels: number[];
  probabilities: number[];
  graph: ModelGraph;
  layers: LayerSnapshot[];
};

export type RunEvent = {
  event_id: string;
  schema_version: string;
  ts_ns: number;
  source: "training" | "runtime_hook" | "checkpoint" | "infra" | "plugin" | "animation";
  type: "metric" | "layer_snapshot" | "infra" | "checkpoint" | "animation" | "run_complete";
  run_id: string;
  session_id?: string | null;
  step: number;
  epoch?: number | null;
  layer?: string | null;
  payload: Record<string, unknown>;
};

export async function getHealth(): Promise<{ status: string; product: string }> {
  const response = await fetch("/health");
  if (!response.ok) throw new Error("Backend health check failed");
  return response.json();
}

export async function getDemoModel(): Promise<ModelGraph> {
  const response = await fetch("/api/demo/model");
  if (!response.ok) throw new Error("Failed to load demo model");
  return response.json();
}

export async function getDemoForward(index: number): Promise<PredictionResponse> {
  const response = await fetch(`/api/demo/forward?index=${index}`);
  if (!response.ok) throw new Error("Failed to run trusted demo forward");
  return response.json();
}

export async function inspectFile(file: File): Promise<InspectionResponse> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/inspect/upload", { method: "POST", body: form });
  if (!response.ok) throw new Error("Failed to inspect .pt file");
  return response.json();
}

export function openDemoStream(onEvent: (event: RunEvent) => void): EventSource {
  const source = new EventSource("/api/runs/demo/stream");
  const eventTypes: RunEvent["type"][] = ["metric", "layer_snapshot", "infra", "checkpoint", "animation", "run_complete"];
  for (const type of eventTypes) {
    source.addEventListener(type, (message) => {
      onEvent(JSON.parse((message as MessageEvent).data) as RunEvent);
    });
  }
  return source;
}
