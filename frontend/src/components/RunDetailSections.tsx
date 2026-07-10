import { Activity, Layers, RefreshCw } from "lucide-react";
import type { RunDetail, RunReport } from "../api/client";
import {
  asMetricText,
  asNumber,
  buildRunTimeline,
  formatConfidence,
  formatMetricValue,
  formatShape,
  metricHighlights,
  metricRecordKeys,
  overviewCards
} from "./RunDetailShared";

type OverviewProps = {
  detail: RunDetail;
  report: RunReport | undefined;
  lastRefreshAt: number | undefined;
  refreshing: boolean;
  onRefresh: () => void;
};

type MetricsProps = {
  detail: RunDetail;
};

type LayersProps = {
  detail: RunDetail;
  report: RunReport | undefined;
  reportLoading: boolean;
};

type EventsProps = {
  detail: RunDetail;
};

export function RunDetailOverviewView({ detail, report, lastRefreshAt, refreshing, onRefresh }: OverviewProps) {
  const heroCards = overviewCards(detail, report, lastRefreshAt);

  return (
    <div className="detail-stack">
      <section className="detail-section">
        <div className="section-heading">
          <div>
            <h3>Overview</h3>
            <p>Current posture first, raw config second.</p>
          </div>
          <button className="detail-mini-action" onClick={onRefresh} type="button">
            <RefreshCw size={14} className={refreshing ? "spin" : ""} /> Refresh
          </button>
        </div>
        <div className="detail-summary-grid">
          {heroCards.map((card) => (
            <article className={`summary-card ${card.tone ?? ""}`} key={card.key}>
              <span>{card.label}</span>
              <strong>{card.value}</strong>
            </article>
          ))}
        </div>
      </section>

      <section className="detail-section">
        <div className="section-heading">
          <div>
            <h3>Provenance</h3>
            <p>Recorded assets that keep replay and reporting trustworthy.</p>
          </div>
        </div>
        <ul className="provenance-list">
          <li data-ok={Boolean(detail.source)}>model source {detail.source ? "recorded" : "missing"}</li>
          <li data-ok={Boolean(detail.graph)}>exact graph {detail.graph ? "recorded" : "missing"}</li>
          <li data-ok={detail.has_samples}>probe samples {detail.has_samples ? "recorded" : "missing"}</li>
          <li data-ok={detail.checkpoints.length > 0}>
            {detail.checkpoints.length} checkpoint{detail.checkpoints.length === 1 ? "" : "s"}
          </li>
        </ul>
      </section>

      <details className="detail-config-disclosure">
        <summary>
          <span>Config</span>
          <small>{detail.config ? `${Object.keys(detail.config).length} recorded values` : "not recorded"}</small>
        </summary>
        {detail.config ? (
          <dl className="config-grid">
            {Object.entries(detail.config).map(([key, value]) => (
              <div key={key}>
                <dt>{key}</dt>
                <dd>{String(value)}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="empty-hint">No config was recorded for this run.</p>
        )}
      </details>
    </div>
  );
}

export function RunDetailMetricsView({ detail }: MetricsProps) {
  const highlights = metricHighlights(detail);
  const metricColumns = metricRecordKeys(detail.metrics).slice(0, 6);
  const recentMetrics = detail.metrics.slice().reverse().slice(0, 12);

  return (
    <div className="detail-stack">
      <section className="detail-section">
        <div className="section-heading">
          <div>
            <h3>Metrics</h3>
            <p>Latest live telemetry and the recent step history.</p>
          </div>
        </div>
        {highlights.length > 0 ? (
          <div className="detail-summary-grid compact">
            {highlights.map((card) => (
              <article className="summary-card" key={card.key}>
                <span>{card.label}</span>
                <strong>{card.value}</strong>
              </article>
            ))}
          </div>
        ) : (
          <p className="empty-hint">No metric snapshots recorded yet.</p>
        )}
      </section>

      {recentMetrics.length > 0 && (
        <section className="detail-section">
          <div className="section-heading">
            <div>
              <h3>Recent steps</h3>
              <p>Most recent snapshots first, based on recorded detail payloads.</p>
            </div>
          </div>
          <div className="table-shell">
            <table className="report-table metrics-table">
              <thead>
                <tr>
                  <th>step</th>
                  <th>epoch</th>
                  {metricColumns.map((key) => (
                    <th key={key}>{key}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {recentMetrics.map((metric, index) => (
                  <tr key={`${metric.step ?? index}-${index}`}>
                    <td>{formatMetricValue(asNumber(metric.step))}</td>
                    <td>{formatMetricValue(asNumber(metric.epoch))}</td>
                    {metricColumns.map((key) => (
                      <td key={key}>{asMetricText(metric[key]) ?? "–"}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

export function RunDetailLayersView({ detail, report, reportLoading }: LayersProps) {
  const graphNodes = detail.graph?.nodes ?? [];
  const graphEdges = detail.graph?.edges ?? [];
  const latestCheckpoint = detail.checkpoints[detail.checkpoints.length - 1];

  return (
    <div className="detail-stack">
      <section className="detail-section">
        <div className="section-heading">
          <div>
            <h3>Layers</h3>
            <p>Graph coverage first, deeper health analysis when a report is ready.</p>
          </div>
        </div>
        <div className="detail-summary-grid compact">
          <article className="summary-card">
            <span>nodes</span>
            <strong>{graphNodes.length}</strong>
          </article>
          <article className="summary-card">
            <span>edges</span>
            <strong>{graphEdges.length}</strong>
          </article>
          <article className="summary-card">
            <span>entry class</span>
            <strong>{detail.entry_class ?? "–"}</strong>
          </article>
          <article className="summary-card">
            <span>latest checkpoint</span>
            <strong>{latestCheckpoint ? `step ${latestCheckpoint.step}` : "–"}</strong>
          </article>
        </div>
      </section>

      {reportLoading && <p className="empty-hint">Analyzing layer health…</p>}

      {report?.layer_health.length ? (
        <section className="detail-section" id="report-layer-health">
          <div className="section-heading">
            <div>
              <h3>Layer Health</h3>
              <p>Derived from recorded run report analysis.</p>
            </div>
          </div>
          <div className="table-shell">
            <table className="report-table">
              <thead>
                <tr>
                  <th>layer</th>
                  <th>sparsity</th>
                  <th>grad norm</th>
                  <th>trend</th>
                  <th>weight drift</th>
                </tr>
              </thead>
              <tbody>
                {report.layer_health.map((layer) => (
                  <tr key={layer.layer_id}>
                    <td>{layer.layer_id}</td>
                    <td>{formatMetricValue(layer.mean_sparsity)}</td>
                    <td>{formatMetricValue(layer.last_gradient_norm)}</td>
                    <td className={`trend-${layer.gradient_trend}`}>{layer.gradient_trend}</td>
                    <td>{formatMetricValue(layer.weight_std_drift)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : (
        !reportLoading && <p className="empty-hint">Layer health appears after report analysis is available.</p>
      )}

      {graphNodes.length > 0 && (
        <section className="detail-section">
          <div className="section-heading">
            <div>
              <h3>Recorded graph</h3>
              <p>Exact node metadata carried with this run detail.</p>
            </div>
          </div>
          <div className="layer-grid">
            {graphNodes.map((node) => (
              <article className="layer-card" key={node.id}>
                <div className="layer-card-header">
                  <strong>{node.label}</strong>
                  <span className={`confidence-badge confidence-${node.confidence}`}>{formatConfidence(node.confidence)}</span>
                </div>
                <dl className="layer-card-meta">
                  <div>
                    <dt>kind</dt>
                    <dd>{node.kind}</dd>
                  </div>
                  <div>
                    <dt>params</dt>
                    <dd>{formatMetricValue(node.param_count)}</dd>
                  </div>
                  <div>
                    <dt>input</dt>
                    <dd>{formatShape(node.input_shape)}</dd>
                  </div>
                  <div>
                    <dt>output</dt>
                    <dd>{formatShape(node.output_shape)}</dd>
                  </div>
                </dl>
              </article>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

export function RunDetailEventLogView({ detail }: EventsProps) {
  const timelineEntries = buildRunTimeline(detail);
  const latestCheckpoint = detail.checkpoints[detail.checkpoints.length - 1];
  const latestMetric = detail.metrics[detail.metrics.length - 1];

  return (
    <div className="detail-stack">
      <section className="detail-section">
        <div className="section-heading">
          <div>
            <h3>Event Log</h3>
            <p>Reliable progressive timeline derived from recorded metrics and checkpoints.</p>
          </div>
        </div>
        <div className="detail-summary-grid compact">
          <article className="summary-card">
            <span>recorded events</span>
            <strong>{detail.event_count}</strong>
          </article>
          <article className="summary-card">
            <span>displayed entries</span>
            <strong>{timelineEntries.length}</strong>
          </article>
          <article className="summary-card">
            <span>latest step</span>
            <strong>{formatMetricValue(asNumber(latestMetric?.step) ?? latestCheckpoint?.step ?? 0)}</strong>
          </article>
          <article className="summary-card">
            <span>last checkpoint</span>
            <strong>{latestCheckpoint ? `step ${latestCheckpoint.step}` : "–"}</strong>
          </article>
        </div>
      </section>

      {timelineEntries.length > 0 ? (
        <section className="event-log-list">
          {timelineEntries.map((entry) => (
            <article className={`event-log-item tone-${entry.tone}`} key={entry.key}>
              <div className="event-log-marker" />
              <div className="event-log-copy">
                <div className="event-log-topline">
                  <strong>{entry.label}</strong>
                  {entry.step != null && <span>step {entry.step}</span>}
                </div>
                <p>{entry.detail}</p>
                {entry.meta && <span className="event-log-meta">{entry.meta}</span>}
              </div>
            </article>
          ))}
        </section>
      ) : (
        <p className="empty-hint">No event snapshots were recorded yet.</p>
      )}
    </div>
  );
}
