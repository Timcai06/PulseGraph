import { Loader2, Play, Radio, RotateCcw, Upload, Zap } from "lucide-react";
import type { RunSummary } from "../api/client";

type Props = {
  onInspect: (file: File) => void;
  onDemoForward: () => void;
  onStartStream: () => void;
  onReset: () => void;
  onWatchRun: (runId: string) => void;
  liveRuns: RunSummary[];
  watchedRunId?: string;
  busy?: "inspect" | "forward";
  streaming: boolean;
  errorMessage?: string;
};

export function ControlRail({
  onInspect,
  onDemoForward,
  onStartStream,
  onReset,
  onWatchRun,
  liveRuns,
  watchedRunId,
  busy,
  streaming,
  errorMessage
}: Props) {
  return (
    <aside className="control-rail">
      <section>
        <h2>Model File</h2>
        <label className={`file-drop ${busy === "inspect" ? "busy" : ""}`}>
          {busy === "inspect" ? <Loader2 size={18} className="spin" /> : <Upload size={18} />}
          <span>{busy === "inspect" ? "Inspecting…" : "Inspect .pt"}</span>
          <input
            type="file"
            accept=".pt,.pth"
            disabled={busy === "inspect"}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onInspect(file);
              event.target.value = "";
            }}
          />
        </label>
        <p className="hint">Default path uses safe weights-only inspection.</p>
        {errorMessage && <p className="error-hint">{errorMessage}</p>}
      </section>

      <section>
        <h2>Trusted Demo</h2>
        <button onClick={onDemoForward} disabled={busy === "forward"}>
          {busy === "forward" ? <Loader2 size={16} className="spin" /> : <Zap size={16} />} Run forward
        </button>
        <button onClick={onStartStream}>
          <Play size={16} /> {streaming && !watchedRunId ? "Restart stream" : "Start stream"}
        </button>
        <button className="secondary" onClick={onReset}>
          <RotateCcw size={16} /> Reset
        </button>
      </section>

      <section>
        <h2>Live Runs</h2>
        {liveRuns.length === 0 && (
          <p className="hint">
            No live runs yet. Start <code>02_train_mlp.py</code> with the backend running to stream real training
            telemetry here.
          </p>
        )}
        {liveRuns.map((run) => (
          <button
            className={`run-item ${watchedRunId === run.run_id ? "watching" : "secondary"}`}
            key={run.run_id}
            onClick={() => onWatchRun(run.run_id)}
            type="button"
          >
            <Radio size={14} className={run.completed ? "" : "live"} />
            <span className="run-name">{run.run_id}</span>
            <span className="run-meta">
              {run.completed ? "done" : "live"} · step {run.last_step}
            </span>
          </button>
        ))}
      </section>
    </aside>
  );
}
