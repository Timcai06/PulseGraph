import { Dumbbell, FileCode2, Info, Loader2, Play, Radio, RotateCcw, Upload, Zap } from "lucide-react";
import type { NamedSourceFile, RunSummary } from "../api/client";

type Props = {
  onInspect: (file: File) => void;
  onSourceImport: (files: NamedSourceFile[]) => void;
  onTrainSource: () => void;
  onRunForward: () => void;
  onStartStream: () => void;
  onReset: () => void;
  onWatchRun: (runId: string) => void;
  onOpenDetail: (runId: string) => void;
  onImportAttach: () => void;
  importAvailable: boolean;
  trainAvailable: boolean;
  forwardTargetLabel: string;
  currentRunKind?: string;
  metricCount: number;
  eventCount: number;
  hasPrediction: boolean;
  liveRuns: RunSummary[];
  watchedRunId?: string;
  busy?: "inspect" | "source" | "train" | "forward";
  streaming: boolean;
  errorMessage?: string;
};

function toNamedFiles(list: FileList | null): NamedSourceFile[] {
  if (!list) return [];
  return Array.from(list)
    .filter((file) => file.name.endsWith(".py") || file.name.endsWith(".zip"))
    .map((file) => ({ file, path: file.name }));
}

export function ControlRail({
  onInspect,
  onSourceImport,
  onTrainSource,
  onRunForward,
  onStartStream,
  onReset,
  onWatchRun,
  onOpenDetail,
  onImportAttach,
  importAvailable,
  trainAvailable,
  forwardTargetLabel,
  currentRunKind,
  metricCount,
  eventCount,
  hasPrediction,
  liveRuns,
  watchedRunId,
  busy,
  streaming,
  errorMessage
}: Props) {
  const streamLabel =
    currentRunKind === "source-import"
      ? "Replay events"
      : currentRunKind === "source-training" || currentRunKind === "recorded-training"
        ? "Stream training"
        : "Demo stream";
  const streamAction = streaming ? `Restart ${streamLabel.toLowerCase()}` : streamLabel;

  return (
    <aside className="control-rail">
      <section>
        <h2>Python Source</h2>
        <label className={`file-drop primary-drop ${busy === "source" ? "busy" : ""}`}>
          {busy === "source" ? <Loader2 size={18} className="spin" /> : <FileCode2 size={18} />}
          <span>{busy === "source" ? "Creating run…" : "Import .py / .zip"}</span>
          <input
            type="file"
            multiple
            accept=".py,.zip"
            disabled={busy === "source"}
            onChange={(event) => {
              const files = toNamedFiles(event.target.files);
              if (files.length) onSourceImport(files);
              event.target.value = "";
            }}
          />
        </label>
        <button onClick={onTrainSource} disabled={!trainAvailable || busy === "train"} type="button">
          {busy === "train" ? <Loader2 size={16} className="spin" /> : <Dumbbell size={16} />} Train source
        </button>
        <p className="hint">Upload source, then run forward or train a short local recipe.</p>
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
        <h2>Weights File</h2>
        <label className={`file-drop compact-drop ${busy === "inspect" ? "busy" : ""}`}>
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
        <p className="hint">Secondary path: inspect weights, then attach source when provenance is missing.</p>
        {importAvailable && (
          <button onClick={onImportAttach} type="button">
            <FileCode2 size={16} /> Attach source for replay
          </button>
        )}
        {importAvailable && (
          <p className="hint">No recorded provenance for these weights — attach the model source to make it replayable.</p>
        )}
      </section>

      <section>
        <h2>Forward Probe</h2>
        <button onClick={onRunForward} disabled={busy === "forward"}>
          {busy === "forward" ? <Loader2 size={16} className="spin" /> : <Zap size={16} />} Run forward
        </button>
        <button onClick={onStartStream}>
          <Play size={16} /> {streamAction}
        </button>
        <button className="secondary" onClick={onReset}>
          <RotateCcw size={16} /> Reset
        </button>
      </section>

      <section>
        <h2>Live Runs</h2>
        {liveRuns.length === 0 && (
          <p className="hint">
            No active training runs. Completed runs are in History.
          </p>
        )}
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
