import { AlertTriangle, CheckCircle2, ChevronDown, CircleDot, Save } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import type { RunEvent } from "../api/client";

type Props = {
  error?: string;
  events: RunEvent[];
  children?: ReactNode;
};

function isDiagnostic(event: RunEvent) {
  if (event.type === "checkpoint" || event.type === "run_complete") return true;
  if (event.type !== "run_status") return false;
  return ["queued", "loading", "building", "preparing_data", "initializing", "checkpointing", "failed", "cancelled"].includes(String(event.payload.phase));
}

function eventLabel(event: RunEvent) {
  if (event.type === "checkpoint") return `Checkpoint saved at step ${event.step}`;
  if (event.type === "run_complete") return `Run ${String(event.payload.status ?? "completed")}`;
  if (event.type === "run_status") return String(event.payload.message ?? event.payload.phase ?? "Run status changed");
  return event.type.replace(/_/g, " ");
}

function EventIcon({ event }: { event: RunEvent }) {
  if (event.type === "checkpoint") return <Save size={13} />;
  if (event.type === "run_complete") return <CheckCircle2 size={13} />;
  return <CircleDot size={13} />;
}

export function DiagnosticsTray({ children, error, events }: Props) {
  const [open, setOpen] = useState(false);
  const diagnostics = events.filter(isDiagnostic).slice(0, 12);
  const latest = error ?? (diagnostics[0] ? eventLabel(diagnostics[0]) : "No warnings or failures");

  useEffect(() => {
    if (error) setOpen(true);
  }, [error]);

  return (
    <details className={`diagnostics-tray ${error ? "has-error" : ""}`} open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>
        <span className="diagnostics-title">
          {error ? <AlertTriangle size={14} /> : <CheckCircle2 size={14} />}
          Diagnostics
        </span>
        <span className="diagnostics-latest">{latest}</span>
        <span className="diagnostics-count">{error ? "1 warning" : `${diagnostics.length} lifecycle`}</span>
        <ChevronDown className="diagnostics-caret" size={14} />
      </summary>
      <div className="diagnostics-list">
        {children}
        {error ? <div className="diagnostic-row warning"><AlertTriangle size={13} /><span>{error}</span></div> : null}
        {diagnostics.map((event) => (
          <div className="diagnostic-row" key={event.event_id}>
            <EventIcon event={event} />
            <span>{eventLabel(event)}</span>
            <time>step {event.step}</time>
          </div>
        ))}
        {!error && diagnostics.length === 0 ? <p>No lifecycle changes or diagnostics yet.</p> : null}
      </div>
    </details>
  );
}
