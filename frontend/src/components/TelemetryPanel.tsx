import { useState, type ReactNode } from "react";
import { Activity, Cpu, Gauge, Square, Timer } from "lucide-react";
import type { MetricSchema } from "../api/client";
import type { MetricPoint, RunProgress, StreamStatus } from "../hooks/useRunStream";
import type { Theme } from "../lib/chartTheme";
import type { MetricGroup } from "../lib/metricSeries";
import { MetricChart } from "./Charts";

type Props = {
  points: MetricPoint[];
  progress?: RunProgress;
  runId?: string;
  runKind?: string;
  status: StreamStatus;
  task?: string;
  metricSchema?: MetricSchema | null;
  selectedStep?: number;
  theme: Theme;
  timeline: ReactNode;
  cancelling?: boolean;
  onCancel?: () => void;
};

const groups: Array<{ value: MetricGroup; label: string }> = [
  { value: "optimization", label: "Optimization" },
  { value: "quality", label: "Quality" },
  { value: "infra", label: "Infra" }
];

function duration(seconds?: number) {
  if (seconds == null || !Number.isFinite(seconds)) return "--";
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds % 60)}s`;
}

function percent(progress?: number) {
  if (progress == null || !Number.isFinite(progress)) return 0;
  return Math.max(0, Math.min(100, progress <= 1 ? progress * 100 : progress));
}

export function TelemetryPanel({
  points,
  progress,
  runId,
  runKind,
  status,
  task,
  metricSchema,
  selectedStep,
  theme,
  timeline,
  cancelling = false,
  onCancel
}: Props) {
  const [group, setGroup] = useState<MetricGroup>("optimization");
  const latest = points[points.length - 1];
  const step = progress?.step ?? latest?.step ?? 0;
  const total = progress?.totalSteps;
  const completion = percent(progress?.progress ?? (total ? step / total : undefined));
  const phase = progress?.phase ?? (status === "streaming" ? "connecting" : status);
  const throughput = latest?.values?.samples_per_sec;

  return (
    <div className="metric-panel">
      <div className="telemetry-heading">
        <div className="panel-heading">
          <div>
            <span className={`telemetry-phase phase-${phase}`}><Activity size={12} /> {phase}</span>
            <h2>Training Telemetry</h2>
          </div>
          <div className="telemetry-heading-actions">
            <span className="telemetry-status-message" title={`${progress?.message ?? "No active run"}${runId ? ` · ${runId}` : ""}`}>
              {progress?.message ?? runId ?? "No active run"}
            </span>
            {status === "streaming" && onCancel ? (
              <button className="telemetry-cancel" disabled={cancelling} onClick={onCancel} type="button">
                <Square size={11} /> {cancelling ? "Cancelling" : "Cancel"}
              </button>
            ) : null}
          </div>
        </div>

        <div className="telemetry-vitals" aria-label="training status">
          <span><Gauge size={13} /><b>{total ? `${step}/${total}` : `step ${step}`}</b></span>
          <span><Timer size={13} /><b>{duration(progress?.elapsedSec)}</b><small>elapsed</small></span>
          <span><Timer size={13} /><b>{duration(progress?.etaSec)}</b><small>ETA</small></span>
          <span><Cpu size={13} /><b>{throughput == null ? "--" : throughput.toFixed(1)}</b><small>samples/s</small></span>
        </div>

        <div className="telemetry-progress" aria-label={`${completion.toFixed(0)}% complete`}>
          <span style={{ transform: `scaleX(${completion / 100})` }} />
        </div>

      </div>

      <div className="telemetry-chart-layout">
        <div className="telemetry-groups" role="tablist" aria-label="Telemetry metric groups">
          {groups.map((item) => (
            <button
              aria-selected={group === item.value}
              className={group === item.value ? "active" : ""}
              key={item.value}
              onClick={() => setGroup(item.value)}
              role="tab"
              type="button"
            >
              {item.label}
            </button>
          ))}
        </div>
        <MetricChart
          group={group}
          points={points}
          status={status}
          theme={theme}
          runKind={runKind}
          task={task}
          metricSchema={metricSchema}
          selectedStep={selectedStep}
        />
      </div>
      {timeline}
    </div>
  );
}
