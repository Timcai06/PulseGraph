import { Loader2, Play } from "lucide-react";
import type { RunDetail } from "../api/client";
import { formatMetricValue } from "./RunDetailShared";
import { SourceAttach } from "./SourceAttach";

type Props = {
  runId: string;
  detail: RunDetail;
  canReplay: boolean;
  replaying: number | undefined;
  onReplay: (step: number) => void;
  onRefreshDetail: () => void;
};

export function RunDetailArtifactsView({
  runId,
  detail,
  canReplay,
  replaying,
  onReplay,
  onRefreshDetail
}: Props) {
  return (
    <div className="detail-stack">
      <section className="detail-section">
        <div className="section-heading">
          <div>
            <h3>Artifacts</h3>
            <p>Source, checkpoints, and replayability collected in one place.</p>
          </div>
        </div>
        <div className="detail-summary-grid compact">
          <article className="summary-card">
            <span>source</span>
            <strong>{detail.source ? "available" : "missing"}</strong>
          </article>
          <article className="summary-card">
            <span>checkpoint count</span>
            <strong>{detail.checkpoints.length}</strong>
          </article>
          <article className="summary-card">
            <span>probe samples</span>
            <strong>{detail.has_samples ? "recorded" : "missing"}</strong>
          </article>
          <article className="summary-card">
            <span>replay</span>
            <strong>{canReplay ? "ready" : "needs source"}</strong>
          </article>
        </div>
      </section>

      <section className="detail-section">
        <div className="section-heading">
          <div>
            <h3>
              {detail.entry_class ?? "Model"} source
              {detail.source_origin && <em className="origin-tag">{detail.source_origin}</em>}
            </h3>
            <p>Attached source stays here so replay and provenance remain operator-visible.</p>
          </div>
        </div>
        {detail.source ? (
          <>
            {detail.source_files.length > 0 && <p className="hint">files: {detail.source_files.join(", ")}</p>}
            <pre className="source-view">{detail.source}</pre>
          </>
        ) : (
          <SourceAttach runId={runId} onAttached={onRefreshDetail} />
        )}
      </section>

      <section className="detail-section">
        <div className="section-heading">
          <div>
            <h3>Checkpoints</h3>
            <p>Replay from any saved checkpoint without leaving the centered detail workspace.</p>
          </div>
        </div>
        {detail.checkpoints.length === 0 && <p className="empty-hint">No checkpoints recorded.</p>}
        {detail.checkpoints.map((checkpoint) => (
          <div className="checkpoint-row" key={checkpoint.step}>
            <div>
              <strong>step {checkpoint.step}</strong>
              <span>
                {checkpoint.epoch != null ? `epoch ${checkpoint.epoch} · ` : ""}
                {formatMetricValue(checkpoint.size_mb, "MB")}
                {checkpoint.fingerprint ? ` · ${checkpoint.fingerprint.slice(0, 10)}` : ""}
              </span>
            </div>
            <button
              disabled={!canReplay || replaying !== undefined}
              onClick={() => onReplay(checkpoint.step)}
              type="button"
            >
              {replaying === checkpoint.step ? <Loader2 size={14} className="spin" /> : <Play size={14} />} Replay
            </button>
          </div>
        ))}
        {!canReplay && detail.checkpoints.length > 0 && <p className="hint">Replay needs recorded model source.</p>}
      </section>
    </div>
  );
}
