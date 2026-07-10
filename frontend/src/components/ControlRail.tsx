import { useEffect, useRef, useState } from "react";
import {
  Dumbbell,
  FileCheck2,
  FileCode2,
  FolderOpen,
  Info,
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
import { inferenceSampleCaption, resolveDetectionOverlay } from "../lib/inferenceView";
import { motionDuration, motionEase } from "../lib/motion";
import { ImagePreview } from "./ImagePreview";

gsap.registerPlugin(useGSAP);

type Props = {
  onResourceUpload: (files: NamedSourceFile[]) => void;
  loadedResource?: LoadedResourceSummary;
  onRunTraining: () => void;
  onRunForward: () => void;
  onReset: () => void;
  onWatchRun: (runId: string) => void;
  onOpenDetail: (runId: string) => void;
  trainAvailable: boolean;
  trainingSteps: number;
  onTrainingStepsChange: (steps: number) => void;
  telemetryStride: number;
  onTelemetryStrideChange: (stride: number) => void;
  forwardTargetLabel: string;
  currentRunKind?: string;
  metricCount: number;
  eventCount: number;
  hasPrediction: boolean;
  liveRuns: RunSummary[];
  watchedRunId?: string;
  busy?: "resource" | "train" | "forward";
  errorMessage?: string;
};

function toNamedFiles(list: FileList | null): NamedSourceFile[] {
  if (!list) return [];
  return Array.from(list)
    .map((file) => {
      const path = (file.webkitRelativePath || file.name).replace(/\\/g, "/");
      return { file, path, folderFile: Boolean(file.webkitRelativePath) };
    })
    .filter(({ path, folderFile }) => {
      const lower = path.toLowerCase();
      return lower.endsWith(".py") || (!folderFile && lower.endsWith(".zip"));
    })
    .map(({ file, path }) => ({ file, path }));
}

function textValue(value: unknown): string | undefined {
  if (typeof value === "string") return value.trim() ? value : undefined;
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (Array.isArray(value) && value.length) {
    return value
      .map((item) => (typeof item === "string" || typeof item === "number" ? String(item) : undefined))
      .filter(Boolean)
      .join(", ");
  }
  return undefined;
}

function resourceContractRows(resource?: LoadedResourceSummary): { label: string; value: string }[] {
  if (!resource) return [];
  const datasetSpec = resource.datasetSpec;
  const outputSchema = resource.outputSchema;
  const metricSchema = resource.metricSchema;
  const rows = [
    { label: "Task", value: resource.task ?? textValue(outputSchema?.kind) },
    { label: "Dataset", value: textValue(datasetSpec?.name) ?? textValue(datasetSpec?.source) ?? resource.dataSource },
    { label: "Output", value: textValue(outputSchema?.renderer) ?? textValue(outputSchema?.kind) },
    { label: "Metric", value: textValue(metricSchema?.primary) ?? textValue(metricSchema?.monitors) }
  ];
  return rows.flatMap((row) => (row.value ? [{ label: row.label, value: row.value }] : []));
}

export function ControlRail({
  onResourceUpload,
  loadedResource,
  onRunTraining,
  onRunForward,
  onReset,
  onWatchRun,
  onOpenDetail,
  trainAvailable,
  trainingSteps,
  onTrainingStepsChange,
  telemetryStride,
  onTelemetryStrideChange,
  forwardTargetLabel,
  currentRunKind,
  metricCount,
  eventCount,
  hasPrediction,
  liveRuns,
  watchedRunId,
  busy,
  errorMessage
}: Props) {
  const [railDrawerOpen, setRailDrawerOpen] = useState(false);
  const drawerRef = useRef<HTMLElement | null>(null);
  const handleRef = useRef<HTMLButtonElement | null>(null);
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    const input = folderInputRef.current;
    if (!input) return;
    input.setAttribute("directory", "");
    input.setAttribute("webkitdirectory", "");
  }, []);

  const contractRows = resourceContractRows(loadedResource);
  const outputRendererHint = textValue(loadedResource?.outputSchema?.renderer) ?? textValue(loadedResource?.outputSchema?.kind);

  useGSAP(
    () => {
      const drawer = drawerRef.current;
      const handle = handleRef.current;
      if (!drawer || !handle) return;

      const handleWidth = handle.offsetWidth || 42;
      const x = railDrawerOpen ? 0 : -(drawer.offsetWidth - handleWidth + drawer.offsetLeft);
      gsap.killTweensOf(drawer);

      if (reducedMotion) {
        gsap.set(drawer, { x });
        return;
      }

      gsap.to(drawer, { x, duration: motionDuration("drawer", reducedMotion), ease: motionEase.panel });
    },
    { dependencies: [railDrawerOpen, reducedMotion], scope: drawerRef }
  );

  return (
    <aside className={`left-control-drawer ${railDrawerOpen ? "open" : ""}`} aria-label="control drawer" ref={drawerRef}>
      <div className="control-rail" id="control-rail-content">
        <section>
          <h2>Training Resource</h2>
          <label className={`file-drop primary-drop ${busy === "resource" ? "busy" : ""} ${loadedResource ? "loaded" : ""}`}>
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
                  {loadedResource.fileCount} file{loadedResource.fileCount === 1 ? "" : "s"} · click to replace
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
          {!loadedResource && (
            <label className={`file-drop compact-drop folder-drop ${busy === "resource" ? "busy" : ""}`}>
              <FolderOpen size={17} />
              <span>Import Folder</span>
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
          )}
          {loadedResource && (
            <>
              <p className="drop-meta">
                {loadedResource.inputShape?.length ? loadedResource.inputShape.join("×") : "?"}
                {" → "}
                {loadedResource.classes ?? "?"} classes
                {loadedResource.dataSource ? ` · ${loadedResource.dataSource}` : ""}
              </p>
              {contractRows.length ? (
                <dl className="resource-contract-grid" aria-label="resource task contract">
                  {contractRows.map((row) => (
                    <div key={row.label}>
                      <dt>{row.label}</dt>
                      <dd title={row.value}>{row.value}</dd>
                    </div>
                  ))}
                </dl>
              ) : null}
              {loadedResource.samples?.length ? (
                <div className="resource-samples" aria-label="resource preview samples">
                  {loadedResource.samples.slice(0, 8).map((sample) => {
                    const detection = resolveDetectionOverlay(sample.output, {
                      task: sample.task,
                      rendererHint: outputRendererHint
                    });
                    const caption = inferenceSampleCaption({
                      output: sample.output,
                      task: sample.task,
                      rendererHint: outputRendererHint,
                      label: sample.label,
                      labelName: sample.label_name
                    });

                    return (
                      <figure key={sample.index}>
                        <ImagePreview
                          pixels={sample.image_pixels}
                          imageShape={sample.image_shape}
                          size="mini"
                          detection={detection}
                          overlayLabel={`Resource sample ${sample.index} detections`}
                        />
                        <figcaption>{caption}</figcaption>
                      </figure>
                    );
                  })}
                </div>
              ) : null}
            </>
          )}
          <label className="number-field">
            <span>Training Steps</span>
            <input
              type="number"
              min={1}
              max={500}
              step={10}
              value={trainingSteps}
              onChange={(event) => onTrainingStepsChange(Number(event.target.value))}
            />
          </label>
          <label className="number-field">
            <span>Telemetry Stride</span>
            <input
              type="number"
              min={1}
              max={500}
              step={1}
              value={telemetryStride}
              onChange={(event) => onTelemetryStrideChange(Number(event.target.value))}
            />
          </label>
          <button onClick={onRunTraining} disabled={!trainAvailable || busy === "train"} type="button">
            {busy === "train" ? <Loader2 size={16} className="spin" /> : <Dumbbell size={16} />} Run Training
          </button>
          {errorMessage && <p className="error-hint">{errorMessage}</p>}
        </section>

        <section className="session-card">
          <div>
            <span>Current run</span>
            <strong>{currentRunKind ?? "demo"}</strong>
          </div>
          <dl>
            <div>
              <dt>target</dt>
              <dd>{forwardTargetLabel}</dd>
            </div>
            <div>
              <dt>events</dt>
              <dd>{eventCount}</dd>
            </div>
            <div>
              <dt>metrics</dt>
              <dd>{metricCount}</dd>
            </div>
            <div>
              <dt>forward</dt>
              <dd>{hasPrediction ? "ready" : "waiting"}</dd>
            </div>
          </dl>
        </section>

        <section>
          <h2>Inference</h2>
          <button onClick={onRunForward} disabled={busy === "forward"}>
            {busy === "forward" ? <Loader2 size={16} className="spin" /> : <Zap size={16} />} Run Inference
          </button>
          <button className="secondary" onClick={onReset}>
            <RotateCcw size={16} /> Reset
          </button>
        </section>

        <section>
          <h2>Live Runs</h2>
          {liveRuns.map((run) => (
            <div className="run-row" key={run.run_id}>
              <button
                className={`run-item ${watchedRunId === run.run_id ? "watching" : "secondary"}`}
                onClick={() => onWatchRun(run.run_id)}
                type="button"
              >
                <Radio size={14} className={run.completed ? "" : "live"} />
                <span className="run-name">{run.run_id}</span>
                <span className="run-meta">
                  {run.completed ? "done" : "live"} · step {run.last_step}
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
        </section>
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
