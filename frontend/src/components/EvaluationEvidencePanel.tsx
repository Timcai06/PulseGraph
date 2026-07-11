import { AlertTriangle, Eye, ScanSearch } from "lucide-react";
import { useState } from "react";
import type { RunDetail } from "../api/client";

type Failure = {
  sampleId: string;
  label: number;
  prediction: number;
  confidence: number;
  probeIndex?: number;
};

type Props = {
  detail?: RunDetail;
  onInspectSample: (index: number) => Promise<void>;
};

function number(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function failuresFromDetail(detail?: RunDetail): { rows: Failure[]; evaluated: number } {
  const evidence = detail?.evidence.slice().reverse().find((item) => item.kind === "evaluation_failures");
  const rawFailures = Array.isArray(evidence?.failures) ? evidence.failures : [];
  const rows = rawFailures.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const row = item as Record<string, unknown>;
    const label = number(row.label);
    const prediction = number(row.prediction);
    const confidence = number(row.confidence);
    if (typeof row.sample_id !== "string" || label == null || prediction == null || confidence == null) return [];
    return [{
      sampleId: row.sample_id,
      label,
      prediction,
      confidence,
      probeIndex: number(row.probe_index)
    }];
  });
  return { rows, evaluated: number(evidence?.evaluated_samples) ?? 0 };
}

function className(detail: RunDetail | undefined, index: number) {
  const names = detail?.config?.class_names;
  return Array.isArray(names) && typeof names[index] === "string" ? names[index] : `class ${index}`;
}

export function EvaluationEvidencePanel({ detail, onInspectSample }: Props) {
  const [inspecting, setInspecting] = useState<number | undefined>();
  const failures = failuresFromDetail(detail);

  const inspect = async (index: number) => {
    setInspecting(index);
    try {
      await onInspectSample(index);
    } finally {
      setInspecting(undefined);
    }
  };

  return (
    <section className="evaluation-evidence-panel">
      <header>
        <div>
          <span><ScanSearch size={13} /> Recorded Evidence</span>
          <h2>Failure Samples</h2>
        </div>
        <em>{failures.rows.length} / {failures.evaluated}</em>
      </header>
      {failures.rows.length ? (
        <div className="failure-sample-list">
          {failures.rows.map((failure) => (
            <article key={failure.sampleId}>
              <div className="failure-sample-title">
                <AlertTriangle size={13} />
                <strong>{failure.sampleId}</strong>
                <em>{(failure.confidence * 100).toFixed(1)}%</em>
              </div>
              <div className="failure-class-path">
                <span>{className(detail, failure.label)}</span>
                <i>→</i>
                <span>{className(detail, failure.prediction)}</span>
              </div>
              {failure.probeIndex != null ? (
                <button
                  disabled={inspecting != null}
                  onClick={() => void inspect(failure.probeIndex as number)}
                  type="button"
                >
                  <Eye size={13} /> {inspecting === failure.probeIndex ? "Loading" : "Inspect"}
                </button>
              ) : null}
            </article>
          ))}
        </div>
      ) : (
        <div className="evaluation-evidence-empty">
          <ScanSearch size={18} />
          <span>No recorded failure evidence</span>
        </div>
      )}
    </section>
  );
}
