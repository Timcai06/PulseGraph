import { Check, Download, FileCode2, FlaskConical, Link, Loader2 } from "lucide-react";
import type { RunDetail, RunReport } from "../api/client";
import { ImagePreview } from "./ImagePreview";
import {
  DetectionEvidenceCard,
  displayClassName,
  formatMetricValue,
  reportSummaryCards,
  severityClass
} from "./RunDetailShared";

type Props = {
  detail: RunDetail;
  report: RunReport | undefined;
  reportLoading: boolean;
  exporting: boolean;
  copied: boolean;
  selectedConfusion: { label: number; prediction: number } | undefined;
  onClearConfusion: () => void;
  onSelectConfusion: (next: { label: number; prediction: number }) => void;
  onDownloadReport: () => void;
  onOpenPrintableReport: (printAfterLoad?: boolean) => void;
  onCopyReportLink: () => void;
};

export function RunReportView({
  detail,
  report,
  reportLoading,
  exporting,
  copied,
  selectedConfusion,
  onClearConfusion,
  onSelectConfusion,
  onDownloadReport,
  onOpenPrintableReport,
  onCopyReportLink
}: Props) {
  const cards = reportSummaryCards(detail, report);
  const hasClassificationAnalysis = Boolean(report?.error_analysis && report.error_analysis.labels.length > 0);
  const hasDetectionEvidence = Boolean(
    report?.detection_analysis && (report.detection_analysis.evidence.length > 0 || report.detection_analysis.evaluated_samples > 0)
  );
  const misclassified = report?.error_analysis?.misclassified ?? [];
  const filteredMisclassified = selectedConfusion
    ? misclassified.filter(
        (sample) => sample.label === selectedConfusion.label && sample.prediction === selectedConfusion.prediction
      )
    : misclassified;

  return (
    <div className="detail-stack">
      <section className="detail-section">
        <div className="section-heading">
          <div>
            <h3>Report</h3>
            <p>Actions are grouped here, but export/share capability stays intact.</p>
          </div>
        </div>
        <div className="report-actions">
          <details className="report-export-menu">
            <summary><Download size={14} /> Export</summary>
            <div>
              <button onClick={onDownloadReport} disabled={exporting} type="button">
                {exporting ? <Loader2 size={14} className="spin" /> : <Download size={14} />} Download report
              </button>
              <button onClick={() => onOpenPrintableReport()} type="button">
                <FileCode2 size={14} /> Printable HTML
              </button>
              <button onClick={() => onOpenPrintableReport(true)} type="button">
                <FlaskConical size={14} /> Print / PDF
              </button>
              <button onClick={onCopyReportLink} type="button">
                {copied ? <Check size={14} /> : <Link size={14} />} Copy link
              </button>
            </div>
          </details>
        </div>
      </section>

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
            {cards.map((card) => (
              <div key={card.key}>
                <span>{card.label}</span>
                <strong>{card.value}</strong>
              </div>
            ))}
          </section>

          <section className="detail-section" id="report-insights">
            <div className="section-heading">
              <div>
                <h3>Insights</h3>
                <p>Focused findings from the generated run report.</p>
              </div>
            </div>
            {report.insights.length > 0 ? (
              report.insights.map((insight) => (
                <div className={`insight ${severityClass(insight.severity)}`} key={insight.title}>
                  <strong>{insight.title}</strong>
                  <p>{insight.detail}</p>
                  {insight.suggestion && <p className="suggestion">→ {insight.suggestion}</p>}
                </div>
              ))
            ) : (
              <p className="empty-hint">No report insights were generated.</p>
            )}
          </section>

          {report.layer_health.length > 0 && (
            <section className="detail-section" id="report-layer-health">
              <div className="section-heading">
                <div>
                  <h3>Layer Health</h3>
                  <p>Same analysis exposed in Layers, kept here for report continuity.</p>
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
          )}

          {report.checkpoint_evaluations.length > 0 && (
            <section className="detail-section" id="report-checkpoints">
              <div className="section-heading">
                <div>
                  <h3>Checkpoints</h3>
                  <p>Probe evaluation at recorded checkpoint steps.</p>
                </div>
              </div>
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
            <section className="detail-section" id="report-error-analysis">
              <div className="section-heading">
                <div>
                  <h3>Error analysis</h3>
                  <p>Confusion drilldown stays intact for selected failure bands.</p>
                </div>
              </div>
              {selectedConfusion && (
                <button className="confusion-clear" onClick={onClearConfusion} type="button">
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
                          <td key={columnIndex} className={count > 0 ? (rowIndex === columnIndex ? "diag" : "confused") : ""}>
                            {count ? (
                              <button
                                className="confusion-cell"
                                onClick={() =>
                                  onSelectConfusion({
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
            <section className="detail-section" id="report-detection-evidence">
              <div className="section-heading">
                <div>
                  <h3>Detection evidence</h3>
                  <p>Rendered with the shared overlay path for checkpoint evidence.</p>
                </div>
              </div>
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
    </div>
  );
}
