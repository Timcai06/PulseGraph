import { Activity, Boxes, CheckCircle2, Database, GitCommitHorizontal, Layers3, ScanSearch, XCircle } from "lucide-react";
import type { RunDetail } from "../api/client";
import { evaluationSnapshot, topFailureRoutes } from "../lib/evaluationSummary";
import { metricHighlights } from "./RunDetailShared";

type Props = {
  detail: RunDetail;
};

function record(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined;
}

function text(value: unknown) {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function count(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

export function EvaluationOverview({ detail }: Props) {
  const snapshot = evaluationSnapshot(detail);
  const successful = Math.max(0, snapshot.evaluated - snapshot.failures.length);
  const successRate = snapshot.evaluated ? successful / snapshot.evaluated : undefined;
  const dataset = record(detail.config?.dataset_spec);
  const classNames = Array.isArray(detail.config?.class_names) ? detail.config.class_names : [];
  const latestCheckpoint = detail.checkpoints[detail.checkpoints.length - 1];
  const highlights = metricHighlights(detail).slice(0, 4);
  const routes = topFailureRoutes(detail);
  const context = [
    { label: "Model", value: text(detail.config?.model) ?? text(detail.entry_class) ?? "recorded" },
    { label: "Dataset", value: text(dataset?.name) ?? "recorded dataset" },
    { label: "Split", value: snapshot.split ?? "evaluation" },
    { label: "Classes", value: String((count(dataset?.classes) ?? classNames.length) || "–") },
    { label: "Checkpoint", value: latestCheckpoint ? `step ${latestCheckpoint.step}` : "none" },
    { label: "Source", value: detail.source_origin === "user-attached" ? "attached" : "recorded" }
  ];

  return (
    <div className="evaluation-overview">
      <div className="evaluation-vitals" aria-label="evaluation summary">
        <span><Database size={14} /><b>{snapshot.evaluated || "–"}</b><small>evaluated</small></span>
        <span><XCircle size={14} /><b>{snapshot.failures.length}</b><small>failures</small></span>
        <span><CheckCircle2 size={14} /><b>{successRate == null ? "–" : `${(successRate * 100).toFixed(1)}%`}</b><small>success</small></span>
        <span><GitCommitHorizontal size={14} /><b>{detail.checkpoints.length}</b><small>checkpoints</small></span>
      </div>

      <div className="evaluation-overview-grid">
        <section className="evaluation-context-band">
          <header><Layers3 size={14} /><span>Run contract</span></header>
          <dl>
            {context.map((item) => <div key={item.label}><dt>{item.label}</dt><dd title={item.value}>{item.value}</dd></div>)}
          </dl>
        </section>

        <section className="evaluation-metric-band">
          <header><Activity size={14} /><span>Latest telemetry</span></header>
          <div>
            {highlights.length ? highlights.map((metric) => (
              <article key={metric.key}><small>{metric.label}</small><strong>{metric.value}</strong></article>
            )) : <em>No recorded metrics</em>}
          </div>
        </section>

        <section className="evaluation-route-band">
          <header><ScanSearch size={14} /><span>Top error routes</span></header>
          <div>
            {routes.length ? routes.map((route) => (
              <article key={`${route.label}-${route.prediction}`}>
                <span>{route.label}</span><Boxes size={12} /><span>{route.prediction}</span><b>{route.count}</b>
              </article>
            )) : <em>No recorded class errors</em>}
          </div>
        </section>
      </div>
    </div>
  );
}
