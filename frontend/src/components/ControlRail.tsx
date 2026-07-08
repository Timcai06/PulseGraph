import { Dumbbell, FileCode2, Info, Loader2, Radio, RotateCcw, Zap } from "lucide-react";
import type { NamedSourceFile, RunSummary } from "../api/client";

type Props = {
  onResourceUpload: (files: NamedSourceFile[]) => void;
  onRunTraining: () => void;
  onRunForward: () => void;
  onReset: () => void;
  onWatchRun: (runId: string) => void;
  onOpenDetail: (runId: string) => void;
  trainAvailable: boolean;
  trainingSteps: number;
  onTrainingStepsChange: (steps: number) => void;
  forwardTargetLabel: string;
  currentRunKind?: string;
  metricCount: number;
  eventCount: number;
  hasPrediction: boolean;
  liveRuns: RunSummary[];
  watchedRunId?: string;
  busy?: "resource" | "train" | "forward";
  errorMessage?: string;
};

function toNamedFiles(list: FileList | null): NamedSourceFile[] {
  if (!list) return [];
  return Array.from(list)
    .filter((file) => file.name.endsWith(".py") || file.name.endsWith(".zip"))
    .map((file) => ({ file, path: file.name }));
}

export function ControlRail({
  onResourceUpload,
  onRunTraining,
  onRunForward,
  onReset,
  onWatchRun,
  onOpenDetail,
  trainAvailable,
  trainingSteps,
  onTrainingStepsChange,
  forwardTargetLabel,
  currentRunKind,
  metricCount,
  eventCount,
  hasPrediction,
  liveRuns,
  watchedRunId,
  busy,
  errorMessage
}: Props) {
  return (
    <aside className="control-rail">
      <section>
        <h2>Training Resource</h2>
        <label className={`file-drop primary-drop ${busy === "resource" ? "busy" : ""}`}>
          {busy === "resource" ? <Loader2 size={18} className="spin" /> : <FileCode2 size={18} />}
          <span>{busy === "resource" ? "Loading…" : "Import .py / .zip"}</span>
          <input
            type="file"
            multiple
            accept=".py,.zip"
            disabled={busy === "resource"}
            onChange={(event) => {
              const files = toNamedFiles(event.target.files);
              if (files.length) onResourceUpload(files);
              event.target.value = "";
            }}
          />
        </label>
        <label className="number-field">
          <span>Training Steps</span>
          <input
            type="number"
            min={1}
            max={500}
            step={10}
            value={trainingSteps}
            onChange={(event) => onTrainingStepsChange(Number(event.target.value))}
          />
        </label>
        <button onClick={onRunTraining} disabled={!trainAvailable || busy === "train"} type="button">
          {busy === "train" ? <Loader2 size={16} className="spin" /> : <Dumbbell size={16} />} Run Training
        </button>
        {errorMessage && <p className="error-hint">{errorMessage}</p>}
      </section>

      <section className="session-card">
        <div>
          <span>Current run</span>
          <strong>{currentRunKind ?? "demo"}</strong>
        </div>
        <dl>
          <div>
            <dt>target</dt>
            <dd>{forwardTargetLabel}</dd>
          </div>
          <div>
            <dt>events</dt>
            <dd>{eventCount}</dd>
          </div>
          <div>
            <dt>metrics</dt>
            <dd>{metricCount}</dd>
          </div>
          <div>
            <dt>forward</dt>
            <dd>{hasPrediction ? "ready" : "waiting"}</dd>
          </div>
        </dl>
      </section>

      <section>
        <h2>Inference</h2>
        <button onClick={onRunForward} disabled={busy === "forward"}>
          {busy === "forward" ? <Loader2 size={16} className="spin" /> : <Zap size={16} />} Run Inference
        </button>
        <button className="secondary" onClick={onReset}>
          <RotateCcw size={16} /> Reset
        </button>
      </section>

      <section>
        <h2>Live Runs</h2>
        {liveRuns.map((run) => (
          <div className="run-row" key={run.run_id}>
            <button
              className={`run-item ${watchedRunId === run.run_id ? "watching" : "secondary"}`}
              onClick={() => onWatchRun(run.run_id)}
              type="button"
            >
              <Radio size={14} className={run.completed ? "" : "live"} />
              <span className="run-name">{run.run_id}</span>
              <span className="run-meta">
                {run.completed ? "done" : "live"} · step {run.last_step}
              </span>
            </button>
            <button
              className="icon-button"
              onClick={() => onOpenDetail(run.run_id)}
              type="button"
              aria-label={`Open ${run.run_id} detail and report`}
              title="Detail & report"
            >
              <Info size={14} />
            </button>
          </div>
        ))}
      </section>
    </aside>
  );
}
