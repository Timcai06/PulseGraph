import type { RunDetail } from "../api/client";

export type EvaluationFailure = {
  sampleId: string;
  label: number;
  prediction: number;
  confidence: number;
  probeIndex?: number;
};

export type EvaluationSnapshot = {
  evaluated: number;
  failures: EvaluationFailure[];
  split?: string;
};

function finiteNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

export function evaluationSnapshot(detail?: RunDetail): EvaluationSnapshot {
  const evidence = detail?.evidence.slice().reverse().find((item) => item.kind === "evaluation_failures");
  const rawFailures = Array.isArray(evidence?.failures) ? evidence.failures : [];
  const failures = rawFailures.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const row = item as Record<string, unknown>;
    const label = finiteNumber(row.label);
    const prediction = finiteNumber(row.prediction);
    const confidence = finiteNumber(row.confidence);
    if (typeof row.sample_id !== "string" || label == null || prediction == null || confidence == null) return [];
    return [{
      sampleId: row.sample_id,
      label,
      prediction,
      confidence,
      probeIndex: finiteNumber(row.probe_index)
    }];
  });

  return {
    evaluated: finiteNumber(evidence?.evaluated_samples) ?? 0,
    failures,
    split: typeof evidence?.split === "string" ? evidence.split : undefined
  };
}

export function evaluationClassName(detail: RunDetail | undefined, index: number) {
  const names = detail?.config?.class_names;
  return Array.isArray(names) && typeof names[index] === "string" ? names[index] : `class ${index}`;
}

export function topFailureRoutes(detail?: RunDetail, limit = 4) {
  const counts = new Map<string, { label: string; prediction: string; count: number }>();
  for (const failure of evaluationSnapshot(detail).failures) {
    const label = evaluationClassName(detail, failure.label);
    const prediction = evaluationClassName(detail, failure.prediction);
    const key = `${label}\u0000${prediction}`;
    const current = counts.get(key);
    counts.set(key, { label, prediction, count: (current?.count ?? 0) + 1 });
  }
  return [...counts.values()].sort((left, right) => right.count - left.count || left.label.localeCompare(right.label)).slice(0, limit);
}
