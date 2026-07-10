import type { PredictionResponse } from "../api/types";

export type ProbabilityRow = {
  index: number;
  label: string;
  value: number;
};

export type ClassificationView = {
  label: number;
  prediction: number;
  confidence: number;
  probabilities: number[];
  classNames?: string[] | null;
};

export type StructuredOutputRow = {
  key: string;
  value: string;
};

export type DetectionBoxView = {
  index: number;
  label: number;
  labelName: string;
  score?: number;
  coordinates: [number, number, number, number];
};

export type DetectionView = {
  boxes: DetectionBoxView[];
  totalCount: number;
  truncated: boolean;
};

export type InferenceRendererView =
  | {
      renderer: "classification";
      kind: string;
      classification: ClassificationView;
    }
  | {
      renderer: "detection";
      kind: string;
      detection: DetectionView;
    }
  | {
      renderer: "structured";
      kind: string;
      rows: StructuredOutputRow[];
    };

type RendererKey = InferenceRendererView["renderer"];

type RendererContext = {
  prediction?: PredictionResponse;
  output?: unknown;
  task?: string | null;
  rendererHint?: string | null;
  kind: string;
};

type RendererResolver = (context: RendererContext) => InferenceRendererView | undefined;

const rendererAliases: Record<string, RendererKey> = {
  classification: "classification",
  probability_chart: "classification",
  detection: "detection",
  box_overlay: "detection"
};

const MAX_RENDERED_DETECTIONS = 100;

function shapeProduct(shape: number[]) {
  return shape.reduce((total, dim) => total * dim, 1);
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined;
}

function numericArray(value: unknown): number[] | undefined {
  return Array.isArray(value) && value.every((item) => typeof item === "number") ? value : undefined;
}

function stringArray(value: unknown): string[] | undefined {
  return Array.isArray(value) && value.every((item) => typeof item === "string") ? value : undefined;
}

function numericBoxArray(value: unknown): [number, number, number, number][] | undefined {
  if (!Array.isArray(value)) return undefined;
  const boxes = value.map((entry) => {
    if (!Array.isArray(entry) || entry.length !== 4 || entry.some((item) => typeof item !== "number")) return undefined;
    return [entry[0], entry[1], entry[2], entry[3]] as [number, number, number, number];
  });
  return boxes.every(Boolean) ? (boxes as [number, number, number, number][]) : undefined;
}

function semanticCandidates(output: unknown, task?: string | null, rendererHint?: string | null): RendererKey[] {
  const record = asRecord(output);
  const values = [record?.kind, record?.renderer, rendererHint, task];
  const seen = new Set<RendererKey>();
  const candidates: RendererKey[] = [];

  for (const value of values) {
    if (typeof value !== "string") continue;
    const semantic = rendererAliases[value.trim().toLowerCase()];
    if (!semantic || seen.has(semantic)) continue;
    seen.add(semantic);
    candidates.push(semantic);
  }

  return candidates;
}

export function displayClassName(index: number, classNames?: string[] | null): string {
  const value = classNames?.[index];
  return value ? value : String(index);
}

export function topProbabilityRows(probabilities: number[], classNames?: string[] | null): ProbabilityRow[] {
  return probabilities
    .map((value, index) => ({ index, label: displayClassName(index, classNames), value }))
    .sort((left, right) => right.value - left.value)
    .slice(0, 3);
}

export function chartProbabilityRows(probabilities: number[], classNames?: string[] | null): ProbabilityRow[] {
  const rows = probabilities.map((value, index) => ({ index, label: displayClassName(index, classNames), value }));
  if (rows.length <= 20) return rows;
  return [...rows].sort((left, right) => right.value - left.value).slice(0, 10);
}

export function inferenceOutputKind(prediction?: PredictionResponse): string {
  const output = asRecord(prediction?.output);
  const kind = output?.kind;
  return typeof kind === "string" && kind ? kind : prediction?.task || "classification";
}

export function classificationOutputFromPrediction(prediction?: PredictionResponse): ClassificationView | undefined {
  if (!prediction) return undefined;
  const output = asRecord(prediction.output);
  const kind = output?.kind;
  if (typeof kind === "string" && kind !== "classification") return undefined;

  const probabilities = numericArray(output?.probabilities) ?? prediction.probabilities ?? [];
  const classNames = stringArray(output?.class_names) ?? prediction.class_names;
  const outputPrediction = output?.prediction;
  const resolvedPrediction =
    typeof outputPrediction === "number" ? outputPrediction : typeof prediction.prediction === "number" ? prediction.prediction : undefined;
  const outputLabel = output?.label;
  const resolvedLabel = typeof outputLabel === "number" ? outputLabel : typeof prediction.label === "number" ? prediction.label : resolvedPrediction;
  if (resolvedPrediction === undefined || resolvedLabel === undefined) return undefined;
  const outputConfidence = output?.confidence;
  const confidence =
    typeof outputConfidence === "number"
      ? outputConfidence
      : probabilities[resolvedPrediction] ?? (probabilities.length ? Math.max(...probabilities) : 0);

  return {
    label: resolvedLabel,
    prediction: resolvedPrediction,
    confidence,
    probabilities,
    classNames
  };
}

export function detectionOutputFromOutput(output: unknown): DetectionView | undefined {
  const record = asRecord(output);
  if (!record) return undefined;
  if (!Array.isArray(record.boxes) || !Array.isArray(record.labels) || record.boxes.length !== record.labels.length) {
    return undefined;
  }
  const displayCount = Math.min(record.boxes.length, MAX_RENDERED_DETECTIONS);
  const boxes = numericBoxArray(record.boxes.slice(0, displayCount));
  const labels = numericArray(record.labels.slice(0, displayCount));
  if (!boxes || !labels) return undefined;
  const scores = Array.isArray(record.scores) ? numericArray(record.scores.slice(0, displayCount)) ?? [] : [];
  const labelNames = Array.isArray(record.label_names) ? stringArray(record.label_names.slice(0, displayCount)) ?? [] : [];
  const declaredTotal = typeof record.total_detections === "number" && Number.isFinite(record.total_detections)
    ? Math.max(0, Math.trunc(record.total_detections))
    : record.boxes.length;
  const totalCount = Math.max(declaredTotal, record.boxes.length);

  return {
    boxes: boxes.map((coordinates, index) => ({
      index,
      label: labels[index],
      labelName: labelNames[index] ?? displayClassName(labels[index]),
      score: typeof scores[index] === "number" ? scores[index] : undefined,
      coordinates
    })),
    totalCount,
    truncated: record.truncated === true || totalCount > boxes.length
  };
}

export function formatConfidencePercent(value: number): string {
  return `${(value * 100).toFixed(value >= 0.995 ? 0 : 1)}%`;
}

export function describeDetectionSummary(detection: DetectionView): string {
  if (!detection.totalCount) return "No objects detected";
  if (detection.totalCount === 1 && detection.boxes.length === 1) {
    const [box] = detection.boxes;
    return box.score === undefined ? box.labelName : `${box.labelName} · ${formatConfidencePercent(box.score)}`;
  }

  const uniqueLabels = [...new Set(detection.boxes.map((box) => box.labelName))];
  if (detection.truncated) return `${detection.boxes.length} shown of ${detection.totalCount} objects`;
  return uniqueLabels.length === 1 ? `${uniqueLabels[0]} ×${detection.totalCount}` : `${detection.totalCount} objects`;
}

export function formatDetectionCoordinates([x1, y1, x2, y2]: [number, number, number, number]): string {
  return `${Math.round(x1)},${Math.round(y1)} to ${Math.round(x2)},${Math.round(y2)}`;
}

export function inferenceSampleCaption({
  output,
  task,
  rendererHint,
  label,
  labelName
}: {
  output?: unknown;
  task?: string | null;
  rendererHint?: string | null;
  label?: number | null;
  labelName?: string | null;
}): string {
  const detection = resolveDetectionOverlay(output, { task, rendererHint });
  if (detection) {
    if (!detection.totalCount) return "No objects";
    if (detection.totalCount === 1 && detection.boxes.length === 1) return detection.boxes[0].labelName;
    return detection.truncated ? `${detection.boxes.length}/${detection.totalCount} objects` : `${detection.totalCount} objects`;
  }

  if (labelName) return labelName;
  if (typeof label === "number") return String(label);
  const kind = typeof asRecord(output)?.kind === "string" ? (asRecord(output)?.kind as string) : task;
  return kind && kind.trim() ? kind : "Sample";
}

function formatStructuredValue(value: unknown): string {
  if (Array.isArray(value)) return `${value.length} items`;
  if (value && typeof value === "object") return `${Object.keys(value).length} fields`;
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  if (typeof value === "boolean") return value ? "true" : "false";
  return value == null ? "empty" : String(value);
}

export function structuredOutputRows(output: unknown): StructuredOutputRow[] {
  const record = asRecord(output);
  if (!record) return [];
  return Object.entries(record)
    .filter(([key]) => key !== "kind")
    .slice(0, 4)
    .map(([key, value]) => ({ key, value: formatStructuredValue(value) }));
}

const rendererRegistry: Record<RendererKey, RendererResolver> = {
  classification: ({ kind, prediction }) => {
    const classification = classificationOutputFromPrediction(prediction);
    return classification ? { renderer: "classification", kind, classification } : undefined;
  },
  detection: ({ kind, output }) => {
    const detection = detectionOutputFromOutput(output);
    return detection ? { renderer: "detection", kind, detection } : undefined;
  },
  structured: ({ kind, output }) => ({ renderer: "structured", kind, rows: structuredOutputRows(output) })
};

export function resolveInferenceRenderer(prediction?: PredictionResponse, rendererHint?: string | null): InferenceRendererView | undefined {
  if (!prediction) return undefined;
  const context: RendererContext = {
    prediction,
    output: prediction.output,
    task: prediction.task,
    rendererHint,
    kind: inferenceOutputKind(prediction)
  };

  for (const candidate of semanticCandidates(prediction.output, prediction.task, rendererHint)) {
    const view = rendererRegistry[candidate](context);
    if (view) return view;
  }

  return rendererRegistry.structured(context);
}

export function resolveDetectionOverlay(
  output: unknown,
  options?: { task?: string | null; rendererHint?: string | null }
): DetectionView | undefined {
  const context: RendererContext = {
    output,
    task: options?.task,
    rendererHint: options?.rendererHint,
    kind: typeof asRecord(output)?.kind === "string" ? (asRecord(output)?.kind as string) : options?.task ?? "structured"
  };

  for (const candidate of semanticCandidates(output, options?.task, options?.rendererHint)) {
    if (candidate !== "detection") continue;
    const view = rendererRegistry.detection(context);
    if (view?.renderer === "detection") return view.detection;
  }

  return undefined;
}

export function normalizeImageShape(_shape: number[] | undefined | null, pixelCount: number): [number, number, number] | undefined {
  const shape = _shape?.map((dim) => Number(dim)).filter((dim) => Number.isFinite(dim) && dim > 0);
  if (shape?.length === 3 && (shape[0] === 1 || shape[0] === 3) && shapeProduct(shape) === pixelCount) {
    return [shape[0], shape[1], shape[2]];
  }
  if (shape?.length === 2 && shapeProduct(shape) === pixelCount) {
    return [1, shape[0], shape[1]];
  }
  const side = Math.sqrt(pixelCount);
  return Number.isInteger(side) ? [1, side, side] : undefined;
}
