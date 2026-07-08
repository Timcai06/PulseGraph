import { FileText, Play, Radio } from "lucide-react";
import type { RunSummary } from "../api/client";

type Props = {
  runs: RunSummary[];
  watchedRunId?: string;
  onWatchRun: (runId: string) => void;
  onOpenDetail: (runId: string) => void;
};

function formatTime(seconds: number) {
  if (!seconds) return "unknown";
  return new Date(seconds * 1000).toLocaleString();
}

export function HistoryPage({ runs, watchedRunId, onWatchRun, onOpenDetail }: Props) {
  return (
    <section className="history-page">
      <header className="page-heading">
        <div>
          <h2>Run History</h2>
          <p>Completed source imports and training runs stay here for replay, reports, and checkpoint inspection.</p>
        </div>
        <span>{runs.length} completed runs</span>
      </header>

      {runs.length === 0 ? (
        <div className="history-empty">
          <p>No completed runs yet.</p>
        </div>
      ) : (
        <div className="history-table">
          <div className="history-row history-head">
            <span>Run</span>
            <span>Last Event</span>
            <span>Step</span>
            <span>Events</span>
            <span>Actions</span>
          </div>
          {runs.map((run) => (
            <div className="history-row" key={run.run_id}>
              <strong>{run.run_id}</strong>
              <span>{formatTime(run.last_event_at)}</span>
              <span>{run.last_step}</span>
              <span>{run.event_count}</span>
              <div className="history-actions">
                <button
                  className={watchedRunId === run.run_id ? "watching" : "secondary"}
                  onClick={() => onWatchRun(run.run_id)}
                  type="button"
                >
                  <Play size={14} /> Replay
                </button>
                <button onClick={() => onOpenDetail(run.run_id)} type="button">
                  <FileText size={14} /> Detail
                </button>
                <Radio size={14} className={watchedRunId === run.run_id ? "history-watching" : ""} />
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
