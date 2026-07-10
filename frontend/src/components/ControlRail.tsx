import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import {
  ChevronDown,
  Dumbbell,
  FileCheck2,
  FileCode2,
  FolderOpen,
  Info,
  ListFilter,
  Loader2,
  PanelLeftClose,
  PanelLeftOpen,
  Radio,
  RotateCcw,
  Zap
} from "lucide-react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import type { NamedSourceFile, RunSummary } from "../api/client";
import type { LoadedResourceSummary } from "../App";
import { useReducedMotion } from "../hooks/useReducedMotion";
import { motionDuration, motionEase } from "../lib/motion";
import { NumericField } from "./NumericField";

gsap.registerPlugin(useGSAP);

type Props = {
  onResourceUpload: (files: NamedSourceFile[]) => void;
  loadedResource?: LoadedResourceSummary;
  onRunTraining: () => void;
  onRunForward: () => void;
  onReset: () => void;
  onWatchRun: (runId: string) => void;
  onOpenDetail: (runId: string) => void;
  onViewAllRuns: () => void;
  trainAvailable: boolean;
  trainingSteps: number;
  onTrainingStepsChange: (steps: number) => void;
  telemetryStride: number;
  onTelemetryStrideChange: (stride: number) => void;
  forwardTargetLabel: string;
  currentRunKind?: string;
  hasPrediction: boolean;
  liveRuns: RunSummary[];
  watchedRunId?: string;
  busy?: "resource" | "train" | "forward";
  errorMessage?: string;
};

type ControlView = "resource" | "train" | "run";

const controlViews: Array<{ value: ControlView; label: string }> = [
  { value: "resource", label: "Resource" },
  { value: "train", label: "Train" },
  { value: "run", label: "Run" }
];

const resourceAssetSuffixes = [".py", ".json", ".png", ".jpg", ".jpeg"];

function toNamedFiles(list: FileList | null): NamedSourceFile[] {
  if (!list) return [];
  return Array.from(list)
    .map((file) => {
      const path = (file.webkitRelativePath || file.name).replace(/\\/g, "/");
      return { file, path, folderFile: Boolean(file.webkitRelativePath) };
    })
    .filter(({ path, folderFile }) => {
      const lower = path.toLowerCase();
      if (!folderFile) return lower.endsWith(".py") || lower.endsWith(".zip");
      return resourceAssetSuffixes.some((suffix) => lower.endsWith(suffix));
    })
    .map(({ file, path }) => ({ file, path }));
}

function Disclosure({
  title,
  subtitle,
  defaultOpen,
  children
}: {
  title: string;
  subtitle?: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(Boolean(defaultOpen));

  return (
    <details className="control-disclosure" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>
        <span className="control-disclosure-copy">
          <strong>{title}</strong>
          {subtitle ? <span>{subtitle}</span> : null}
        </span>
        <ChevronDown size={14} className="control-disclosure-caret" />
      </summary>
      <div className="control-disclosure-body">{children}</div>
    </details>
  );
}

function ViewTab({ active, label, onSelect, value, count }: { active: boolean; label: string; onSelect: (view: ControlView) => void; value: ControlView; count?: string }) {
  return (
    <button
      aria-controls={`control-panel-${value}`}
      aria-selected={active}
      className={active ? "active" : ""}
      id={`control-tab-${value}`}
      onClick={() => onSelect(value)}
      role="tab"
      tabIndex={active ? 0 : -1}
      type="button"
    >
      <span className="control-tab-copy">
        <strong>{label}</strong>
      </span>
      {count ? <span className="control-tab-count">{count}</span> : null}
    </button>
  );
}

function ResourceView({
  busy,
  errorMessage,
  loadedResource,
  onResourceUpload
}: Pick<Props, "busy" | "errorMessage" | "loadedResource" | "onResourceUpload">) {
  const folderInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const input = folderInputRef.current;
    if (!input) return;
    input.setAttribute("directory", "");
    input.setAttribute("webkitdirectory", "");
  }, []);

  return (
    <div aria-labelledby="control-tab-resource" className="control-panel" id="control-panel-resource" role="tabpanel">
      <section className="control-card control-card--hero">
        <div className="control-card-heading">
          <div>
            <span className="control-card-kicker">Resource</span>
            <h3>Import</h3>
          </div>
          <span className={`control-card-badge ${busy === "resource" ? "busy" : loadedResource ? "ready" : "idle"}`}>
            {busy === "resource" ? "analyzing" : loadedResource ? "loaded" : "empty"}
          </span>
        </div>

        <label className={`file-drop primary-drop control-upload ${busy === "resource" ? "busy" : ""} ${loadedResource ? "loaded" : ""}`}>
          {busy === "resource" ? (
            <Loader2 size={18} className="spin" />
          ) : loadedResource ? (
            <FileCheck2 size={18} />
          ) : (
            <FileCode2 size={18} />
          )}
          {busy === "resource" ? (
            <span>Analyzing…</span>
          ) : loadedResource ? (
            <span className="drop-loaded">
              <strong>{loadedResource.name}</strong>
              <em>
                {loadedResource.fileCount} file{loadedResource.fileCount === 1 ? "" : "s"}
              </em>
            </span>
          ) : (
            <span>Import .py / .zip</span>
          )}
          <input
            type="file"
            multiple
            accept=".py,.zip"
            disabled={busy === "resource"}
            onChange={(event) => {
              const files = toNamedFiles(event.target.files);
              if (files.length) onResourceUpload(files);
              event.target.value = "";
            }}
          />
        </label>

        <div className="control-inline-actions">
          <label className={`control-mini-action ${busy === "resource" ? "busy" : ""}`}>
            <FolderOpen size={14} />
            <span>Folder</span>
            <input
              type="file"
              multiple
              disabled={busy === "resource"}
              ref={folderInputRef}
              onChange={(event) => {
                const files = toNamedFiles(event.target.files);
                if (files.length) onResourceUpload(files);
                event.target.value = "";
              }}
            />
          </label>
        </div>
      </section>

      {errorMessage ? <p className="error-hint">{errorMessage}</p> : null}
    </div>
  );
}

function TrainView({
  busy,
  errorMessage,
  loadedResource,
  onRunTraining,
  onTelemetryStrideChange,
  onTrainingStepsChange,
  telemetryStride,
  trainAvailable,
  trainingSteps
}: Pick<
  Props,
  | "busy"
  | "errorMessage"
  | "loadedResource"
  | "onRunTraining"
  | "onTelemetryStrideChange"
  | "onTrainingStepsChange"
  | "telemetryStride"
  | "trainAvailable"
  | "trainingSteps"
>) {
  return (
    <div aria-labelledby="control-tab-train" className="control-panel" id="control-panel-train" role="tabpanel">
      <section className="control-card control-card--hero">
        <div className="control-card-heading">
          <div>
            <span className="control-card-kicker">Train</span>
            <h3>{loadedResource?.name ?? "No resource"}</h3>
          </div>
          <span className={`control-card-badge ${trainAvailable ? "ready" : "blocked"}`}>{trainAvailable ? "ready" : "blocked"}</span>
        </div>

        <div className="control-parameter-grid" aria-label="training parameters">
          <NumericField label="Training Steps" onCommit={onTrainingStepsChange} range={{ min: 1, max: 500 }} value={trainingSteps} />
          <NumericField label="Telemetry Stride" onCommit={onTelemetryStrideChange} range={{ min: 1, max: 500 }} value={telemetryStride} />
        </div>

        <button onClick={onRunTraining} disabled={!trainAvailable || busy === "train"} type="button">
          {busy === "train" ? <Loader2 size={16} className="spin" /> : <Dumbbell size={16} />} Run Training
        </button>
      </section>

      {errorMessage ? <p className="error-hint">{errorMessage}</p> : null}
    </div>
  );
}

function RunView({
  busy,
  errorMessage,
  forwardTargetLabel,
  hasPrediction,
  liveRuns,
  onOpenDetail,
  onViewAllRuns,
  onReset,
  onRunForward,
  onWatchRun,
  watchedRunId,
  currentRunKind
}: Pick<
  Props,
  | "busy"
  | "errorMessage"
  | "forwardTargetLabel"
  | "hasPrediction"
  | "liveRuns"
  | "onOpenDetail"
  | "onViewAllRuns"
  | "onReset"
  | "onRunForward"
  | "onWatchRun"
  | "watchedRunId"
  | "currentRunKind"
>) {
  const orderedRuns = useMemo(() => [...liveRuns].sort((a, b) => b.last_step - a.last_step || b.event_count - a.event_count), [liveRuns]);
  const previewRuns = orderedRuns.slice(0, 3);

  return (
    <div aria-labelledby="control-tab-run" className="control-panel" id="control-panel-run" role="tabpanel">
      <section className="control-card control-card--hero">
        <div className="control-card-heading">
          <div>
            <span className="control-card-kicker">Run</span>
            <h3>{currentRunKind ?? "demo run"}</h3>
          </div>
          <span className={`control-card-badge ${hasPrediction ? "ready" : "idle"}`}>{hasPrediction ? "forward ready" : "idle"}</span>
        </div>

        <div className="control-run-context">
          <span>Target</span>
          <strong title={forwardTargetLabel}>{forwardTargetLabel || "none"}</strong>
        </div>

        <div className="control-inline-actions">
          <button onClick={onRunForward} disabled={busy === "forward"} type="button">
            {busy === "forward" ? <Loader2 size={16} className="spin" /> : <Zap size={16} />} Run Inference
          </button>
          <button className="secondary" onClick={onReset} type="button">
            <RotateCcw size={16} /> Reset
          </button>
        </div>
      </section>

      <Disclosure title="Live runs" subtitle={`${orderedRuns.length} active`} defaultOpen={orderedRuns.length > 0 && orderedRuns.length <= 3}>
        {previewRuns.length ? (
          <div className="control-run-list">
            {orderedRuns.length > previewRuns.length ? (
              <button className="control-view-all secondary" onClick={onViewAllRuns} type="button">
                <ListFilter size={14} /> View all {orderedRuns.length}
              </button>
            ) : null}
            {previewRuns.map((run) => (
              <div className="run-row" key={run.run_id}>
                <button
                  className={`run-item ${watchedRunId === run.run_id ? "watching" : "secondary"}`}
                  onClick={() => onWatchRun(run.run_id)}
                  type="button"
                >
                  <Radio size={14} className={run.completed ? "" : "live"} />
                  <span className="run-copy">
                    <span className="run-name">{run.run_id}</span>
                    <span className="run-meta">
                      {run.completed ? "done" : "live"} · step {run.last_step}
                    </span>
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
          </div>
        ) : (
          <p className="control-empty-note">No live runs yet.</p>
        )}
      </Disclosure>

      {errorMessage ? <p className="error-hint">{errorMessage}</p> : null}
    </div>
  );
}

export function ControlRail({
  onResourceUpload,
  loadedResource,
  onRunTraining,
  onRunForward,
  onReset,
  onWatchRun,
  onOpenDetail,
  onViewAllRuns,
  trainAvailable,
  trainingSteps,
  onTrainingStepsChange,
  telemetryStride,
  onTelemetryStrideChange,
  forwardTargetLabel,
  currentRunKind,
  hasPrediction,
  liveRuns,
  watchedRunId,
  busy,
  errorMessage
}: Props) {
  const [railDrawerOpen, setRailDrawerOpen] = useState(true);
  const [activeView, setActiveView] = useState<ControlView>("resource");
  const [drawerWidth, setDrawerWidth] = useState(0);
  const drawerRef = useRef<HTMLElement | null>(null);
  const handleRef = useRef<HTMLButtonElement | null>(null);
  const viewRef = useRef<HTMLDivElement | null>(null);
  const viewScrollPositionsRef = useRef<Record<ControlView, number>>({ resource: 0, train: 0, run: 0 });
  const reducedMotion = useReducedMotion();

  const selectControlView = (nextView: ControlView, focus = false) => {
    if (nextView === activeView) return;
    const viewport = viewRef.current;
    if (viewport) viewScrollPositionsRef.current[activeView] = viewport.scrollTop;
    setActiveView(nextView);
    requestAnimationFrame(() => {
      viewRef.current?.scrollTo({ top: viewScrollPositionsRef.current[nextView], behavior: "auto" });
      if (focus) document.getElementById(`control-tab-${nextView}`)?.focus();
    });
  };

  useEffect(() => {
    if (!busy) return;
    if (busy === "resource") selectControlView("resource");
    if (busy === "train") selectControlView("train");
    if (busy === "forward") selectControlView("run");
  }, [busy]);

  useEffect(() => {
    const drawer = drawerRef.current;
    if (!drawer || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => setDrawerWidth(Math.round(entry.contentRect.width)));
    observer.observe(drawer);
    return () => observer.disconnect();
  }, []);

  useGSAP(
    () => {
      const drawer = drawerRef.current;
      const handle = handleRef.current;
      if (!drawer || !handle) return;

      const handleWidth = handle.offsetWidth || 42;
      const x = railDrawerOpen ? 0 : -(drawer.offsetWidth - handleWidth + drawer.offsetLeft);
      gsap.killTweensOf(drawer, "x");

      if (reducedMotion) {
        gsap.set(drawer, { x });
        return;
      }

      gsap.to(drawer, { x, duration: motionDuration("drawer", reducedMotion), ease: motionEase.panel });
    },
    { dependencies: [drawerWidth, railDrawerOpen, reducedMotion], scope: drawerRef }
  );

  useGSAP(
    () => {
      const panel = viewRef.current?.firstElementChild as HTMLElement | null;
      if (!panel) return;
      if (reducedMotion) {
        gsap.set(panel, { opacity: 1, y: 0 });
        return;
      }
      gsap.fromTo(
        panel,
        { opacity: 0, y: 8 },
        {
          opacity: 1,
          y: 0,
          duration: motionDuration("enter", reducedMotion),
          ease: motionEase.standard,
          clearProps: "opacity,transform"
        }
      );
    },
    { dependencies: [activeView, reducedMotion], scope: drawerRef }
  );

  const tabCounts = {
    resource: loadedResource ? `${loadedResource.fileCount} files` : "empty",
    train: `${trainingSteps} step${trainingSteps === 1 ? "" : "s"}`,
    run: `${liveRuns.length} live`
  } satisfies Record<ControlView, string>;

  const handleTabKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    const currentIndex = controlViews.findIndex((view) => view.value === activeView);
    let nextIndex = currentIndex;
    if (event.key === "ArrowRight") nextIndex = (currentIndex + 1) % controlViews.length;
    else if (event.key === "ArrowLeft") nextIndex = (currentIndex - 1 + controlViews.length) % controlViews.length;
    else if (event.key === "Home") nextIndex = 0;
    else if (event.key === "End") nextIndex = controlViews.length - 1;
    else return;

    event.preventDefault();
    const nextView = controlViews[nextIndex].value;
    selectControlView(nextView, true);
  };

  return (
    <aside className={`left-control-drawer ${railDrawerOpen ? "open" : ""}`} aria-label="control drawer" ref={drawerRef}>
      <div className="control-rail" id="control-rail-content">
        <header className="control-rail-header">
          <div>
            <span className="control-rail-kicker">Workflow</span>
            <h2>Control</h2>
          </div>
          <span className={`control-rail-badge ${busy ? "busy" : loadedResource ? "ready" : "idle"}`}>{busy ? "busy" : activeView}</span>
        </header>

        <nav className="control-tabs" aria-label="Control contexts" onKeyDown={handleTabKeyDown} role="tablist">
          {controlViews.map((view) => (
            <ViewTab
              active={activeView === view.value}
              count={tabCounts[view.value]}
              key={view.value}
              label={view.label}
              onSelect={selectControlView}
              value={view.value}
            />
          ))}
        </nav>

        <div className="control-view-frame" ref={viewRef}>
          {activeView === "resource" ? (
            <ResourceView
              busy={busy}
              errorMessage={errorMessage}
              loadedResource={loadedResource}
              onResourceUpload={onResourceUpload}
            />
          ) : activeView === "train" ? (
            <TrainView
              busy={busy}
              errorMessage={errorMessage}
              loadedResource={loadedResource}
              onRunTraining={onRunTraining}
              onTelemetryStrideChange={onTelemetryStrideChange}
              onTrainingStepsChange={onTrainingStepsChange}
              telemetryStride={telemetryStride}
              trainAvailable={trainAvailable}
              trainingSteps={trainingSteps}
            />
          ) : (
            <RunView
              busy={busy}
              currentRunKind={currentRunKind}
              errorMessage={errorMessage}
              forwardTargetLabel={forwardTargetLabel}
              hasPrediction={hasPrediction}
              liveRuns={liveRuns}
              onOpenDetail={onOpenDetail}
              onViewAllRuns={onViewAllRuns}
              onReset={onReset}
              onRunForward={onRunForward}
              onWatchRun={onWatchRun}
              watchedRunId={watchedRunId}
            />
          )}
        </div>
      </div>

      <button
        aria-controls="control-rail-content"
        aria-expanded={railDrawerOpen}
        aria-label={railDrawerOpen ? "Collapse controls" : "Expand controls"}
        className="rail-drawer-handle"
        onClick={() => setRailDrawerOpen((open) => !open)}
        ref={handleRef}
        type="button"
      >
        {railDrawerOpen ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
        <span>Controls</span>
      </button>
    </aside>
  );
}
