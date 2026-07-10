import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Check, Download, FileCode2, FlaskConical, Layers, Link, Loader2, Play, X } from "lucide-react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import { Flip } from "gsap/Flip";
import type { PredictionResponse, RunDetail, RunReport } from "../api/client";
import {
  downloadRunReportMarkdown,
  getRunDetail,
  getRunReport,
  runForward,
  runReportHtmlUrl,
  runReportMarkdownUrl
} from "../api/client";
import { useReducedMotion } from "../hooks/useReducedMotion";
import { displayClassName, normalizeImageShape, type DetectionView } from "../lib/inferenceView";
import { motionDuration, motionDurations, motionEase } from "../lib/motion";
import { ImagePreview } from "./ImagePreview";
import { SourceAttach } from "./SourceAttach";

gsap.registerPlugin(useGSAP, Flip);

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

type SummaryCard = {
  key: string;
  label: string;
  value: string;
};

type DetectionEvidenceSample = NonNullable<NonNullable<RunReport["detection_analysis"]>["evidence"]>[number];

function formatMetricValue(value: number | string | null | undefined, unit?: string | null) {
  if (value == null) return "–";
  const rendered = typeof value === "number" ? (Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "")) : value;
  return unit ? `${rendered} ${unit}` : rendered;
}

function formatBox(box: number[]) {
  return box.map((value) => value.toFixed(2).replace(/\.00$/, "")).join(", ");
}

function detectionLabel(label: number, labelName?: string | null) {
  return labelName || String(label);
}

function evidenceDetection(sample: DetectionEvidenceSample): DetectionView {
  const totalCount = sample.predicted_total ?? sample.predicted.length;
  return {
    boxes: sample.predicted.flatMap((box, index) =>
      box.box.length === 4
        ? [{
            index,
            label: box.label,
            labelName: detectionLabel(box.label, box.label_name),
            score: box.score ?? undefined,
            coordinates: [box.box[0], box.box[1], box.box[2], box.box[3]] as [number, number, number, number]
          }]
        : []
    ),
    totalCount,
    truncated: sample.predicted_truncated === true || totalCount > sample.predicted.length
  };
}

function DetectionEvidenceCard({ sample }: { sample: DetectionEvidenceSample }) {
  const shape = normalizeImageShape(sample.image_shape, sample.image_pixels.length);
  const imageLabel = shape ? `${shape[2]}×${shape[1]}` : "image";
  const detection = evidenceDetection(sample);

  return (
    <article className="detection-evidence-card">
      <div className="detection-evidence-header">
        <strong>sample {sample.sample_index}</strong>
        <span>mean IoU {formatMetricValue(sample.mean_iou)}</span>
      </div>
      <div className="detection-evidence-body">
        <div className="detection-evidence-preview">
          <ImagePreview
            pixels={sample.image_pixels}
            imageShape={sample.image_shape}
            size="normal"
            detection={detection}
            overlayLabel={`Checkpoint detections for sample ${sample.sample_index}`}
          />
          <span className="detection-evidence-shape">{imageLabel}</span>
        </div>
        <div className="detection-evidence-columns">
          <div>
            <h4>
              Predicted
              {sample.predicted_truncated ? ` ${sample.predicted.length}/${sample.predicted_total}` : ""}
            </h4>
            {sample.predicted.length > 0 ? (
              <table className="report-table detection-box-table">
                <thead>
                  <tr>
                    <th>label</th>
                    <th>score</th>
                    <th>IoU</th>
                    <th>box</th>
                  </tr>
                </thead>
                <tbody>
                  {sample.predicted.map((box, index) => (
                    <tr key={`${sample.sample_index}-pred-${index}`}>
                      <td>{detectionLabel(box.label, box.label_name)}</td>
                      <td>{formatMetricValue(box.score)}</td>
                      <td>{formatMetricValue(box.matched_iou)}</td>
                      <td>{formatBox(box.box)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="empty-hint">No predicted boxes.</p>
            )}
          </div>
          <div>
            <h4>
              Target
              {sample.target_truncated ? ` ${sample.target.length}/${sample.target_total}` : ""}
            </h4>
            {sample.target.length > 0 ? (
              <table className="report-table detection-box-table">
                <thead>
                  <tr>
                    <th>label</th>
                    <th>IoU</th>
                    <th>box</th>
                  </tr>
                </thead>
                <tbody>
                  {sample.target.map((box, index) => (
                    <tr key={`${sample.sample_index}-target-${index}`}>
                      <td>{detectionLabel(box.label, box.label_name)}</td>
                      <td>{formatMetricValue(box.matched_iou)}</td>
                      <td>{formatBox(box.box)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="empty-hint">No target boxes.</p>
            )}
          </div>
        </div>
      </div>
    </article>
  );
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
    gsap.from(overlay, { opacity: 0, duration: motionDurations.enter, ease: motionEase.signalOut });
    const rect = panel.getBoundingClientRect();
    if (origin && rect.width && rect.height) {
      gsap.set(panel, {
        x: origin.left - rect.left,
        y: origin.top - rect.top,
        scaleX: origin.width / rect.width,
        scaleY: origin.height / rect.height,
        transformOrigin: "top left"
      });
      const state = Flip.getState(panel, { props: "transform" });
      gsap.set(panel, { x: 0, y: 0, scaleX: 1, scaleY: 1 });
      Flip.from(state, {
        duration: motionDuration("panel", reducedMotion),
        ease: motionEase.panel,
        absolute: true,
        clearProps: "transform"
      });
    } else {
      gsap.from(panel, {
        scale: 0.96,
        y: 12,
        duration: motionDuration("enter", reducedMotion),
        ease: motionEase.standard,
        clearProps: "transform"
      });
    }
  }, [runId]);
  const [detail, setDetail] = useState<RunDetail | undefined>();
  const [report, setReport] = useState<RunReport | undefined>();
  const [loading, setLoading] = useState(true);
  const [reportLoading, setReportLoading] = useState(false);
  const [replaying, setReplaying] = useState<number | undefined>();
  const [exporting, setExporting] = useState(false);
  const [copied, setCopied] = useState(false);
  const [selectedConfusion, setSelectedConfusion] = useState<{ label: number; prediction: number } | undefined>();
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

  const handleDownloadReport = async () => {
    setExporting(true);
    setError(undefined);
    try {
      const blob = await downloadRunReportMarkdown(runId);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${runId}-report.md`;
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Report export failed.");
    } finally {
      setExporting(false);
    }
  };

  const handleCopyReportLink = async () => {
    setError(undefined);
    try {
      const href = new URL(runReportMarkdownUrl(runId), window.location.origin).toString();
      await navigator.clipboard.writeText(href);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      setError("Could not copy the report link.");
    }
  };

  const handleOpenPrintableReport = (printAfterLoad = false) => {
    setError(undefined);
    const opened = window.open(runReportHtmlUrl(runId), "_blank");
    if (!opened) {
      setError("Could not open the printable report.");
      return;
    }
    if (printAfterLoad) {
      opened.addEventListener("load", () => opened.print(), { once: true });
    }
  };

  const canReplay = Boolean(detail?.source && detail?.checkpoints.length);
  const filteredMisclassified = useMemo(() => {
    const misclassified = report?.error_analysis?.misclassified ?? [];
    if (!selectedConfusion) return misclassified;
    return misclassified.filter(
      (sample) => sample.label === selectedConfusion.label && sample.prediction === selectedConfusion.prediction
    );
  }, [report, selectedConfusion]);
  const reportTask = report?.task ?? String(detail?.config?.task ?? "classification");
  const summaryCards = useMemo<SummaryCard[]>(() => {
    if (!report) return [];
    const cards: SummaryCard[] = [{ key: "task", label: "task", value: reportTask }];
    const seen = new Set(cards.map((card) => card.key));
    const addCard = (key: string, label: string, value: number | string | null | undefined, unit?: string | null) => {
      if (value == null || seen.has(key)) return;
      cards.push({ key, label, value: formatMetricValue(value, unit) });
      seen.add(key);
    };

    report.task_metrics.forEach((metric) => addCard(metric.key, metric.label, metric.value, metric.unit));
    addCard("final_loss", "final loss", report.final_loss);
    if (reportTask === "classification") {
      addCard("best_accuracy", "best accuracy", report.best_accuracy);
      addCard("overfit_gap", "overfit gap", report.overfit_gap);
    }
    addCard("loss_plateau_step", "plateau step", report.loss_plateau_step);
    return cards;
  }, [report, reportTask]);
  const hasClassificationAnalysis = Boolean(report?.error_analysis && report.error_analysis.labels.length > 0);
  const hasDetectionEvidence = Boolean(report?.detection_analysis && (report.detection_analysis.evidence.length > 0 || report.detection_analysis.evaluated_samples > 0));

  return (
    <div className="detail-overlay" role="dialog" aria-label={`Run ${runId} detail`} ref={overlayRef}>
      <div className="detail-panel shared-detail" ref={panelRef}>
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
              <div className="report-actions">
                <button onClick={handleDownloadReport} disabled={exporting} type="button">
                  {exporting ? <Loader2 size={14} className="spin" /> : <Download size={14} />} Download report
                </button>
                <button onClick={() => handleOpenPrintableReport()} type="button">
                  <FileCode2 size={14} /> Printable HTML
                </button>
                <button onClick={() => handleOpenPrintableReport(true)} type="button">
                  <FlaskConical size={14} /> Print / PDF
                </button>
                <button onClick={handleCopyReportLink} type="button">
                  {copied ? <Check size={14} /> : <Link size={14} />} Copy link
                </button>
              </div>
              {reportLoading && <p className="empty-hint">Analyzing recorded signals…</p>}
              {report && (
                <>
                  <nav className="report-nav" aria-label="report sections">
                    <a href="#report-summary">Summary</a>
                    <a href="#report-insights">Insights</a>
                    <a href="#report-layer-health">Layer Health</a>
                    {report.checkpoint_evaluations.length > 0 && <a href="#report-checkpoints">Checkpoints</a>}
                    {hasClassificationAnalysis && <a href="#report-error-analysis">Error Analysis</a>}
                    {hasDetectionEvidence && <a href="#report-detection-evidence">Detection Evidence</a>}
                  </nav>
                  <section className="report-stats" id="report-summary">
                    {summaryCards.map((card) => (
                      <div key={card.key}>
                        <span>{card.label}</span>
                        <strong>{card.value}</strong>
                      </div>
                    ))}
                  </section>
                  <section id="report-insights">
                    <h3>Insights</h3>
                    {report.insights.map((insight) => (
                      <div className={`insight ${severityClass(insight.severity)}`} key={insight.title}>
                        <strong>{insight.title}</strong>
                        <p>{insight.detail}</p>
                        {insight.suggestion && <p className="suggestion">→ {insight.suggestion}</p>}
                      </div>
                    ))}
                  </section>
                  {report.layer_health.length > 0 && (
                    <section id="report-layer-health">
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
                    <section id="report-checkpoints">
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
                    <section id="report-error-analysis">
                      <h3>Error analysis</h3>
                      {selectedConfusion && (
                        <button className="confusion-clear" onClick={() => setSelectedConfusion(undefined)} type="button">
                          Showing {displayClassName(selectedConfusion.label, report.error_analysis?.class_names)} →{" "}
                          {displayClassName(selectedConfusion.prediction, report.error_analysis?.class_names)}
                        </button>
                      )}
                      <div className="confusion-wrap">
                        <table className="report-table confusion">
                          <thead>
                            <tr>
                              <th>true \ pred</th>
                              {report.error_analysis.labels.map((label) => (
                                <th key={label}>{displayClassName(label, report.error_analysis?.class_names)}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {report.error_analysis.confusion.map((row, rowIndex) => (
                              <tr key={rowIndex}>
                                <td>
                                  {displayClassName(
                                    report.error_analysis?.labels[rowIndex] ?? rowIndex,
                                    report.error_analysis?.class_names
                                  )}
                                </td>
                                {row.map((count, columnIndex) => (
                                  <td
                                    key={columnIndex}
                                    className={count > 0 ? (rowIndex === columnIndex ? "diag" : "confused") : ""}
                                  >
                                    {count ? (
                                      <button
                                        className="confusion-cell"
                                        onClick={() =>
                                          setSelectedConfusion({
                                            label: report.error_analysis?.labels[rowIndex] ?? rowIndex,
                                            prediction: report.error_analysis?.labels[columnIndex] ?? columnIndex
                                          })
                                        }
                                        type="button"
                                      >
                                        {count}
                                      </button>
                                    ) : (
                                      ""
                                    )}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      {filteredMisclassified.length > 0 && (
                        <div className="misclassified">
                          {filteredMisclassified.map((sample) => (
                            <div className="missample" key={sample.index}>
                              <ImagePreview pixels={sample.pixels} imageShape={sample.image_shape} size="mini" />
                              <span>
                                {sample.label_name ?? displayClassName(sample.label, report.error_analysis?.class_names)} →{" "}
                                {sample.prediction_name ??
                                  displayClassName(sample.prediction, report.error_analysis?.class_names)}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </section>
                  )}
                  {report.detection_analysis && hasDetectionEvidence && (
                    <section id="report-detection-evidence">
                      <h3>Detection evidence</h3>
                      <div className="detection-evidence-meta">
                        <span>mean IoU {formatMetricValue(report.detection_analysis.mean_iou)}</span>
                        <span>{report.detection_analysis.evaluated_samples} evaluated samples</span>
                      </div>
                      <div className="detection-evidence-grid">
                        {report.detection_analysis.evidence.map((sample) => (
                          <DetectionEvidenceCard key={sample.sample_index} sample={sample} />
                        ))}
                      </div>
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
