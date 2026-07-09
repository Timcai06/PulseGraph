import { useCallback, useEffect, useRef, useState } from "react";
import { FileCode2, FlaskConical, Layers, Loader2, Play, X } from "lucide-react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import type { PredictionResponse, RunDetail, RunReport } from "../api/client";
import { getRunDetail, getRunReport, runForward } from "../api/client";
import { useReducedMotion } from "../hooks/useReducedMotion";
import { ImagePreview } from "./ImagePreview";
import { SourceAttach } from "./SourceAttach";

gsap.registerPlugin(useGSAP);

type Tab = "overview" | "source" | "checkpoints" | "report";

type Props = {
  runId: string;
  initialTab?: Tab;
  /* card rect the panel visually grows out of (shared-element transition) */
  origin?: DOMRect;
  onClose: () => void;
  onPrediction: (prediction: PredictionResponse) => void;
};

function severityClass(severity: string) {
  return severity === "critical" ? "critical" : severity === "warning" ? "warning" : "info";
}

export function RunDetailPanel({ runId, initialTab = "overview", origin, onClose, onPrediction }: Props) {
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const reducedMotion = useReducedMotion();
  const [tab, setTab] = useState<Tab>(initialTab);

  useGSAP(() => {
    const overlay = overlayRef.current;
    const panel = panelRef.current;
    if (!overlay || !panel || reducedMotion) return;
    gsap.from(overlay, { opacity: 0, duration: 0.3, ease: "power1.out" });
    const rect = panel.getBoundingClientRect();
    if (origin && rect.width && rect.height) {
      gsap.from(panel, {
        x: origin.left - rect.left,
        y: origin.top - rect.top,
        scaleX: origin.width / rect.width,
        scaleY: origin.height / rect.height,
        transformOrigin: "top left",
        duration: 0.5,
        ease: "power3.inOut",
        clearProps: "transform"
      });
    } else {
      gsap.from(panel, { scale: 0.96, y: 12, duration: 0.35, ease: "power2.out", clearProps: "transform" });
    }
  }, [runId]);
  const [detail, setDetail] = useState<RunDetail | undefined>();
  const [report, setReport] = useState<RunReport | undefined>();
  const [loading, setLoading] = useState(true);
  const [reportLoading, setReportLoading] = useState(false);
  const [replaying, setReplaying] = useState<number | undefined>();
  const [error, setError] = useState<string | undefined>();

  const refreshDetail = useCallback(() => {
    getRunDetail(runId)
      .then(setDetail)
      .catch(() => setError("Failed to load run detail."))
      .finally(() => setLoading(false));
  }, [runId]);

  useEffect(() => {
    setLoading(true);
    setDetail(undefined);
    setReport(undefined);
    setTab(initialTab);
    refreshDetail();
  }, [refreshDetail, initialTab]);

  useEffect(() => {
    if (tab !== "report" || report) return;
    setReportLoading(true);
    getRunReport(runId)
      .then(setReport)
      .catch(() => setError("Failed to build the report."))
      .finally(() => setReportLoading(false));
  }, [tab, report, runId]);

  const handleReplay = async (step: number) => {
    setReplaying(step);
    setError(undefined);
    try {
      onPrediction(await runForward(runId, step));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Forward replay failed.");
    } finally {
      setReplaying(undefined);
    }
  };

  const canReplay = Boolean(detail?.source && detail?.checkpoints.length);

  return (
    <div className="detail-overlay" role="dialog" aria-label={`Run ${runId} detail`} ref={overlayRef}>
      <div className="detail-panel" ref={panelRef}>
        <header className="detail-header">
          <div>
            <h2>{runId}</h2>
            <span>
              {detail?.completed ? "completed" : "live"} · {detail?.event_count ?? 0} events ·{" "}
              {detail?.checkpoints.length ?? 0} checkpoints
              {detail?.has_samples ? " · probe samples" : ""}
            </span>
          </div>
          <button className="icon-button" onClick={onClose} type="button" aria-label="Close detail">
            <X size={16} />
          </button>
        </header>

        <nav className="detail-tabs">
          <button className={tab === "overview" ? "active" : ""} onClick={() => setTab("overview")} type="button">
            <Layers size={14} /> Overview
          </button>
          <button className={tab === "source" ? "active" : ""} onClick={() => setTab("source")} type="button">
            <FileCode2 size={14} /> Source
          </button>
          <button className={tab === "checkpoints" ? "active" : ""} onClick={() => setTab("checkpoints")} type="button">
            <Play size={14} /> Checkpoints
          </button>
          <button className={tab === "report" ? "active" : ""} onClick={() => setTab("report")} type="button">
            <FlaskConical size={14} /> Report
          </button>
        </nav>

        <div className="detail-body">
          {error && <p className="error-hint">{error}</p>}
          {loading && <p className="empty-hint">Loading run provenance…</p>}

          {!loading && detail && tab === "overview" && (
            <>
              <section>
                <h3>Config</h3>
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
              </section>
              <section>
                <h3>Provenance</h3>
                <ul className="provenance-list">
                  <li data-ok={Boolean(detail.source)}>model source {detail.source ? "recorded" : "missing"}</li>
                  <li data-ok={Boolean(detail.graph)}>exact graph {detail.graph ? "recorded" : "missing"}</li>
                  <li data-ok={detail.has_samples}>probe samples {detail.has_samples ? "recorded" : "missing"}</li>
                  <li data-ok={detail.checkpoints.length > 0}>
                    {detail.checkpoints.length} checkpoint{detail.checkpoints.length === 1 ? "" : "s"}
                  </li>
                </ul>
              </section>
            </>
          )}

          {!loading && detail && tab === "source" && (
            <section>
              <h3>
                {detail.entry_class ?? "Model"} source
                {detail.source_origin && <em className="origin-tag">{detail.source_origin}</em>}
              </h3>
              {detail.source ? (
                <>
                  {detail.source_files.length > 1 && (
                    <p className="hint">files: {detail.source_files.join(", ")} (showing entry file)</p>
                  )}
                  <pre className="source-view">{detail.source}</pre>
                </>
              ) : (
                <SourceAttach runId={runId} onAttached={refreshDetail} />
              )}
            </section>
          )}

          {!loading && detail && tab === "checkpoints" && (
            <section>
              <h3>Checkpoint timeline</h3>
              {detail.checkpoints.length === 0 && <p className="empty-hint">No checkpoints recorded.</p>}
              {detail.checkpoints.map((checkpoint) => (
                <div className="checkpoint-row" key={checkpoint.step}>
                  <div>
                    <strong>step {checkpoint.step}</strong>
                    <span>
                      {checkpoint.epoch != null ? `epoch ${checkpoint.epoch} · ` : ""}
                      {checkpoint.size_mb} MB
                      {checkpoint.fingerprint ? ` · ${checkpoint.fingerprint.slice(0, 10)}` : ""}
                    </span>
                  </div>
                  <button
                    disabled={!canReplay || replaying !== undefined}
                    onClick={() => handleReplay(checkpoint.step)}
                    type="button"
                  >
                    {replaying === checkpoint.step ? <Loader2 size={14} className="spin" /> : <Play size={14} />} Replay
                  </button>
                </div>
              ))}
              {!canReplay && detail.checkpoints.length > 0 && (
                <p className="hint">Replay needs recorded model source.</p>
              )}
            </section>
          )}

          {!loading && tab === "report" && (
            <>
              {reportLoading && <p className="empty-hint">Analyzing recorded signals…</p>}
              {report && (
                <>
                  <section>
                    <h3>Insights</h3>
                    {report.insights.map((insight) => (
                      <div className={`insight ${severityClass(insight.severity)}`} key={insight.title}>
                        <strong>{insight.title}</strong>
                        <p>{insight.detail}</p>
                        {insight.suggestion && <p className="suggestion">→ {insight.suggestion}</p>}
                      </div>
                    ))}
                  </section>
                  <section className="report-stats">
                    <div>
                      <span>final loss</span>
                      <strong>{report.final_loss ?? "–"}</strong>
                    </div>
                    <div>
                      <span>best accuracy</span>
                      <strong>{report.best_accuracy ?? "–"}</strong>
                    </div>
                    <div>
                      <span>overfit gap</span>
                      <strong>{report.overfit_gap ?? "–"}</strong>
                    </div>
                    <div>
                      <span>plateau step</span>
                      <strong>{report.loss_plateau_step ?? "–"}</strong>
                    </div>
                  </section>
                  {report.layer_health.length > 0 && (
                    <section>
                      <h3>Layer health</h3>
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
                              <td>{layer.mean_sparsity ?? "–"}</td>
                              <td>{layer.last_gradient_norm ?? "–"}</td>
                              <td className={`trend-${layer.gradient_trend}`}>{layer.gradient_trend}</td>
                              <td>{layer.weight_std_drift ?? "–"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </section>
                  )}
                  {report.checkpoint_evaluations.length > 0 && (
                    <section>
                      <h3>Checkpoint accuracy (probe samples)</h3>
                      <div className="checkpoint-bars">
                        {report.checkpoint_evaluations.map((evaluation) => (
                          <div className="checkpoint-bar" key={evaluation.step}>
                            <span className="bar-label">step {evaluation.step}</span>
                            <span className="bar-track">
                              <span className="bar-fill" style={{ width: `${(evaluation.accuracy ?? 0) * 100}%` }} />
                            </span>
                            <span className="bar-value">{((evaluation.accuracy ?? 0) * 100).toFixed(0)}%</span>
                          </div>
                        ))}
                      </div>
                    </section>
                  )}
                  {report.error_analysis && report.error_analysis.labels.length > 0 && (
                    <section>
                      <h3>Error analysis</h3>
                      <div className="confusion-wrap">
                        <table className="report-table confusion">
                          <thead>
                            <tr>
                              <th>true \ pred</th>
                              {report.error_analysis.labels.map((label) => (
                                <th key={label}>{label}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {report.error_analysis.confusion.map((row, rowIndex) => (
                              <tr key={rowIndex}>
                                <td>{report.error_analysis?.labels[rowIndex]}</td>
                                {row.map((count, columnIndex) => (
                                  <td
                                    key={columnIndex}
                                    className={count > 0 ? (rowIndex === columnIndex ? "diag" : "confused") : ""}
                                  >
                                    {count || ""}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      {report.error_analysis.misclassified.length > 0 && (
                        <div className="misclassified">
                          {report.error_analysis.misclassified.map((sample) => (
                            <div className="missample" key={sample.index}>
                              <ImagePreview pixels={sample.pixels} size="mini" />
                              <span>
                                {sample.label} → {sample.prediction}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </section>
                  )}
                </>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
