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

  const probabilities = numericArray(output?.probabilities) ?? prediction.probabilities;
  const classNames = stringArray(output?.class_names) ?? prediction.class_names;
  const outputPrediction = output?.prediction;
  const resolvedPrediction = typeof outputPrediction === "number" ? outputPrediction : prediction.prediction;
  const outputLabel = output?.label;
  const resolvedLabel = typeof outputLabel === "number" ? outputLabel : prediction.label;
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
