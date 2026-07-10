import { useCallback, useEffect, useRef, useState } from "react";
import { Activity, Clock3, FlaskConical, Layers, Package, ShieldCheck, X } from "lucide-react";
import gsap from "gsap";
import { Flip } from "gsap/Flip";
import { useGSAP } from "@gsap/react";
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
import { motionDuration, motionDurations, motionEase } from "../lib/motion";
import { RunDetailArtifactsView } from "./RunDetailArtifactsView";
import {
  RunDetailEventLogView,
  RunDetailLayersView,
  RunDetailMetricsView,
  RunDetailOverviewView
} from "./RunDetailSections";
import { latestStep, shouldPollRunDetail } from "./RunDetailShared";
import { RunReportView } from "./RunReportView";

gsap.registerPlugin(useGSAP, Flip);

type Tab = "overview" | "metrics" | "layers" | "artifacts" | "report" | "events";
type InitialTab = Tab | "source";

type Props = {
  runId: string;
  initialTab?: InitialTab;
  origin?: DOMRect;
  onClose: () => void;
  onPrediction: (result: PredictionResponse) => void;
};

const POLL_MS = 1200;

function normalizeTab(tab: InitialTab): Tab {
  return tab === "source" ? "artifacts" : tab;
}

export function RunDetailPanel({ runId, initialTab = "overview", origin, onClose, onPrediction }: Props) {
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const requestRef = useRef(0);
  const copyTimerRef = useRef<number | undefined>();
  const reducedMotion = useReducedMotion();
  const [tab, setTab] = useState<Tab>(() => normalizeTab(initialTab));
  const [detail, setDetail] = useState<RunDetail | undefined>();
  const [report, setReport] = useState<RunReport | undefined>();
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefreshAt, setLastRefreshAt] = useState<number | undefined>();
  const [reportLoading, setReportLoading] = useState(false);
  const [replaying, setReplaying] = useState<number | undefined>();
  const [exporting, setExporting] = useState(false);
  const [copied, setCopied] = useState(false);
  const [selectedConfusion, setSelectedConfusion] = useState<{ label: number; prediction: number } | undefined>();
  const [error, setError] = useState<string | undefined>();
  const reportAvailable = Boolean(detail && (detail.completed || detail.checkpoints.length > 0));

  useGSAP(
    () => {
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
          scale: 0.97,
          y: 12,
          duration: motionDuration("enter", reducedMotion),
          ease: motionEase.standard,
          clearProps: "transform"
        });
      }
    },
    { dependencies: [runId], scope: overlayRef }
  );

  const refreshDetail = useCallback(async (silent = false) => {
    const requestId = ++requestRef.current;
    if (silent) setRefreshing(true);
    else setLoading(true);
    try {
      const next = await getRunDetail(runId);
      if (requestId !== requestRef.current) return;
      setDetail(next);
      setLastRefreshAt(Date.now());
      setError(undefined);
    } catch {
      if (requestId === requestRef.current && !silent) setError("Failed to load run detail.");
    } finally {
      if (requestId === requestRef.current) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, [runId]);

  useEffect(() => {
    setTab(normalizeTab(initialTab));
    setDetail(undefined);
    setReport(undefined);
    setSelectedConfusion(undefined);
    setError(undefined);
    void refreshDetail(false);
    return () => {
      requestRef.current += 1;
    };
  }, [initialTab, refreshDetail]);

  useEffect(() => {
    if (!shouldPollRunDetail(detail)) return;
    const timer = window.setInterval(() => void refreshDetail(true), POLL_MS);
    return () => window.clearInterval(timer);
  }, [detail?.completed, refreshDetail]);

  useEffect(() => {
    if ((tab !== "report" && tab !== "layers") || report || !reportAvailable) return;
    let cancelled = false;
    setReportLoading(true);
    getRunReport(runId)
      .then((next) => {
        if (!cancelled) setReport(next);
      })
      .catch(() => {
        if (!cancelled) setError("Failed to build the report.");
      })
      .finally(() => {
        if (!cancelled) setReportLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [report, reportAvailable, runId, tab]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  useEffect(() => () => {
    if (copyTimerRef.current) window.clearTimeout(copyTimerRef.current);
  }, []);

  const handleReplay = async (step: number) => {
    setReplaying(step);
    setError(undefined);
    try {
      onPrediction(await runForward(runId, step));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Forward replay failed.");
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
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Report export failed.");
    } finally {
      setExporting(false);
    }
  };

  const handleCopyReportLink = async () => {
    setError(undefined);
    try {
      await navigator.clipboard.writeText(new URL(runReportMarkdownUrl(runId), window.location.origin).toString());
      setCopied(true);
      if (copyTimerRef.current) window.clearTimeout(copyTimerRef.current);
      copyTimerRef.current = window.setTimeout(() => setCopied(false), 1200);
    } catch {
      setError("Could not copy the report link.");
    }
  };

  const handleOpenPrintableReport = (printAfterLoad = false) => {
    const opened = window.open(runReportHtmlUrl(runId), "_blank");
    if (!opened) {
      setError("Could not open the printable report.");
      return;
    }
    if (printAfterLoad) opened.addEventListener("load", () => opened.print(), { once: true });
  };

  const canReplay = Boolean(detail?.source && detail.checkpoints.length);
  const currentStep = latestStep(detail);

  return (
    <div
      className="detail-overlay"
      role="dialog"
      aria-modal="true"
      aria-label={`Run ${runId} detail`}
      ref={overlayRef}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="detail-panel shared-detail" ref={panelRef}>
        <header className="detail-header">
          <div className="detail-identity">
            <span className={`detail-live-state ${detail?.completed ? "completed" : "live"}`}>
              {detail?.completed ? "completed" : "live sync"}
            </span>
            <h2>{runId}</h2>
            <p>{detail ? `${detail.event_count} events · ${detail.checkpoints.length} checkpoints` : "Loading run state"}</p>
          </div>
          <div className="detail-status-strip" aria-live="polite">
            <span><small>step</small><strong>{currentStep}</strong></span>
            <span><small>metrics</small><strong>{detail?.metrics.length ?? 0}</strong></span>
            <span><small>sync</small><strong>{refreshing ? "updating" : detail?.completed ? "final" : "live"}</strong></span>
          </div>
          <button className="icon-button" onClick={onClose} type="button" aria-label="Close detail">
            <X size={16} />
          </button>
        </header>

        <nav className="detail-tabs" aria-label="Run detail views">
          <button className={tab === "overview" ? "active" : ""} onClick={() => setTab("overview")} type="button"><ShieldCheck size={14} /> Overview</button>
          <button className={tab === "metrics" ? "active" : ""} onClick={() => setTab("metrics")} type="button"><Activity size={14} /> Metrics</button>
          <button className={tab === "layers" ? "active" : ""} onClick={() => setTab("layers")} type="button"><Layers size={14} /> Layers</button>
          <button className={tab === "artifacts" ? "active" : ""} onClick={() => setTab("artifacts")} type="button"><Package size={14} /> Artifacts</button>
          <button className={tab === "report" ? "active" : ""} onClick={() => setTab("report")} type="button"><FlaskConical size={14} /> Report</button>
          <button className={tab === "events" ? "active" : ""} onClick={() => setTab("events")} type="button"><Clock3 size={14} /> Event Log</button>
        </nav>

        <div className="detail-body">
          {error ? <p className="error-hint">{error}</p> : null}
          {loading && !detail ? <p className="empty-hint">Loading run state…</p> : null}
          {detail && tab === "overview" ? <RunDetailOverviewView detail={detail} report={report} lastRefreshAt={lastRefreshAt} refreshing={refreshing} onRefresh={() => void refreshDetail(true)} /> : null}
          {detail && tab === "metrics" ? <RunDetailMetricsView detail={detail} /> : null}
          {detail && tab === "layers" ? <RunDetailLayersView detail={detail} report={report} reportLoading={reportLoading} /> : null}
          {detail && tab === "artifacts" ? <RunDetailArtifactsView runId={runId} detail={detail} canReplay={canReplay} replaying={replaying} onReplay={(step) => void handleReplay(step)} onRefreshDetail={() => void refreshDetail(false)} /> : null}
          {detail && tab === "report" ? <RunReportView detail={detail} report={report} reportLoading={reportLoading} exporting={exporting} copied={copied} selectedConfusion={selectedConfusion} onClearConfusion={() => setSelectedConfusion(undefined)} onSelectConfusion={setSelectedConfusion} onDownloadReport={() => void handleDownloadReport()} onOpenPrintableReport={handleOpenPrintableReport} onCopyReportLink={() => void handleCopyReportLink()} /> : null}
          {detail && tab === "events" ? <RunDetailEventLogView detail={detail} /> : null}
        </div>
      </div>
    </div>
  );
}
