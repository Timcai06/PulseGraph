import { AlertTriangle, Eye, ScanSearch } from "lucide-react";
import { useState } from "react";
import type { RunDetail } from "../api/client";
import { evaluationClassName, evaluationSnapshot } from "../lib/evaluationSummary";

type Props = {
  detail?: RunDetail;
  onInspectSample: (index: number) => Promise<void>;
};

export function EvaluationEvidencePanel({ detail, onInspectSample }: Props) {
  const [inspecting, setInspecting] = useState<number | undefined>();
  const snapshot = evaluationSnapshot(detail);

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
        <em>{snapshot.failures.length} / {snapshot.evaluated}</em>
      </header>
      {snapshot.failures.length ? (
        <div className="failure-sample-list">
          {snapshot.failures.map((failure) => (
            <article key={failure.sampleId}>
              <div className="failure-sample-title">
                <AlertTriangle size={13} />
                <strong>{failure.sampleId}</strong>
                <em>{(failure.confidence * 100).toFixed(1)}%</em>
              </div>
              <div className="failure-class-path">
                <span>{evaluationClassName(detail, failure.label)}</span>
                <i>→</i>
                <span>{evaluationClassName(detail, failure.prediction)}</span>
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
