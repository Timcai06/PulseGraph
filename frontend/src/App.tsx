import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronUp, Zap } from "lucide-react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import type {
  DatasetSpec,
  GraphNode,
  ImageSample,
  MetricSchema,
  ModelGraph,
  NamedSourceFile,
  OutputSchema,
  PredictionResponse,
  RunSummary
} from "./api/client";
import {
  deleteRun,
  cancelRun,
  getDemoModel,
  getHealth,
  getRunDetail,
  listRuns,
  previewResource,
  runForward,
  trainResourceRun
} from "./api/client";
import { TopStatusBar } from "./components/TopStatusBar";
import { ControlRail } from "./components/ControlRail";
import { ModelGraphPanel } from "./components/ModelGraphPanel";
import { InferenceProbe } from "./components/InferenceProbe";
import { DiagnosticsTray } from "./components/DiagnosticsTray";
import { TelemetryPanel } from "./components/TelemetryPanel";
import { RunDetailPanel } from "./components/RunDetailPanel";
import { HistoryPage } from "./components/HistoryPage";
import { StageStats } from "./components/StageStats";
import { TrainingLoopStrip } from "./components/TrainingLoopStrip";
import { LayerInspector } from "./components/LayerInspector";
import { TimelineScrubber } from "./components/TimelineScrubber";
import { useRunStream } from "./hooks/useRunStream";
import { useReducedMotion } from "./hooks/useReducedMotion";
import type { Theme } from "./lib/chartTheme";
import type { ForwardTarget } from "./lib/forwardTarget";
import { describeForwardTarget, resolveForwardTarget } from "./lib/forwardTarget";
import { firstDisplayNode } from "./lib/graphView";
import type { GhostEdge } from "./lib/graphPorts";
import { classificationOutputFromPrediction, displayClassName, inferenceOutputKind } from "./lib/inferenceView";
import { configureMotionDefaults, motionDuration, motionDurations, motionEase, motionStagger } from "./lib/motion";
import { splitRunBuckets } from "./lib/runViews";
import { runContractFromConfig, type RunContract } from "./lib/runContract";
import {
  deriveCausalFocus,
  deriveTimelineFrames,
  eventsAtTimelineStep,
  layerSnapshotsAtStep,
  resolveTimelineStep
} from "./lib/timeline";
import { deriveTrainingLoopModel, type TrainingLoopStage } from "./lib/trainingLoop";

gsap.registerPlugin(useGSAP);

const emptyGraph: ModelGraph = { nodes: [], edges: [] };
const HEALTH_POLL_MS = 8000;
const RUNS_POLL_MS = 5000;
const THEME_KEY = "pulsegraph-theme";
const DEFAULT_TRAINING_STEPS = 100;
const DEFAULT_TELEMETRY_STRIDE = 5;
// must match --dock-handle-h in base.css
const DOCK_HANDLE_PX = 42;

const sampleSourceLabel: Record<PredictionResponse["sample_source"], string> = {
  mnist: "Real dataset",
  probe: "Resource sample",
  synthetic: "Synthetic probe"
};

function initialTheme(): Theme {
  const saved = window.localStorage.getItem(THEME_KEY);
  return saved === "light" ? "light" : "dark";
}

export type LoadedResourceSummary = {
  name: string;
  fileCount: number;
  task?: string;
  datasetSpec?: DatasetSpec | null;
  outputSchema?: OutputSchema | null;
  metricSchema?: MetricSchema | null;
  inputShape?: number[];
  classes?: number;
  classNames?: string[];
  dataSource?: string;
  samples?: ImageSample[];
};

type SourceRecipe = {
  files: NamedSourceFile[];
  entryFile: string;
  summary?: LoadedResourceSummary;
};

type CurrentRunKind = "resource-training" | "source-training" | "recorded-training";
type Workspace = "prepare" | "train" | "evaluate" | "runs";

const workspaces: Array<{ value: Workspace; label: string }> = [
  { value: "prepare", label: "Prepare" },
  { value: "train", label: "Train" },
  { value: "evaluate", label: "Evaluate" },
  { value: "runs", label: "Runs" }
];

function initialWorkspace(): Workspace {
  const value = window.location.hash.replace(/^#\/?/, "");
  return workspaces.some((workspace) => workspace.value === value) ? value as Workspace : "prepare";
}

function preferredResourceEntry(files: NamedSourceFile[]): string | undefined {
  return files.find((file) => file.path === "resource.py" || file.path.endsWith("/resource.py"))?.path ?? files[0]?.path;
}

export default function App() {
  const [backendStatus, setBackendStatus] = useState("checking");
  const [graph, setGraph] = useState<ModelGraph>(emptyGraph);
  const [selectedNode, setSelectedNode] = useState<GraphNode | undefined>();
  const [inspectedNodeId, setInspectedNodeId] = useState<string | undefined>();
  const [prediction, setPrediction] = useState<PredictionResponse | undefined>();
  const [liveRuns, setLiveRuns] = useState<RunSummary[]>([]);
  const [busy, setBusy] = useState<"resource" | "train" | "forward" | undefined>();
  const [cancellingRunId, setCancellingRunId] = useState<string | undefined>();
  const [errorMessage, setErrorMessage] = useState<string | undefined>();
  const [detailRunId, setDetailRunId] = useState<string | undefined>();
  const [detailInitialTab, setDetailInitialTab] = useState<"overview" | "source">("overview");
  const [detailOrigin, setDetailOrigin] = useState<DOMRect | undefined>();
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [workspace, setWorkspace] = useState<Workspace>(initialWorkspace);
  const [historyMode, setHistoryMode] = useState<"completed" | "live">("completed");
  const [forwardTarget, setForwardTarget] = useState<ForwardTarget | undefined>();
  const [pendingForwardRun, setPendingForwardRun] = useState<string | undefined>();
  const [sourceRecipe, setSourceRecipe] = useState<SourceRecipe | undefined>();
  const [activeRunContract, setActiveRunContract] = useState<RunContract | undefined>();
  const [currentRunKind, setCurrentRunKind] = useState<CurrentRunKind | undefined>();
  const [trainingSteps, setTrainingSteps] = useState(DEFAULT_TRAINING_STEPS);
  const [telemetryStride, setTelemetryStride] = useState(DEFAULT_TELEMETRY_STRIDE);
  const [dockOpen, setDockOpen] = useState(false);
  const [forwardTick, setForwardTick] = useState(0);
  const [selectedTimelineStep, setSelectedTimelineStep] = useState<number | undefined>();
  const [ghostEdges, setGhostEdges] = useState<GhostEdge[]>([]);
  const [selectedGhostEdgeId, setSelectedGhostEdgeId] = useState<string | undefined>();
  const shellRef = useRef<HTMLElement | null>(null);
  const dockRef = useRef<HTMLElement | null>(null);
  const reducedMotion = useReducedMotion();
  const stream = useRunStream();

  useEffect(() => {
    const media = configureMotionDefaults();
    return () => media.revert();
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const selectWorkspace = useCallback((nextWorkspace: Workspace) => {
    setWorkspace(nextWorkspace);
    const nextHash = `#/${nextWorkspace}`;
    if (window.location.hash !== nextHash) window.history.replaceState(null, "", nextHash);
  }, []);

  useEffect(() => {
    const syncWorkspace = () => setWorkspace(initialWorkspace());
    window.addEventListener("hashchange", syncWorkspace);
    if (!window.location.hash) window.history.replaceState(null, "", "#/prepare");
    return () => window.removeEventListener("hashchange", syncWorkspace);
  }, []);

  useGSAP(
    () => {
      if (reducedMotion) return;
      const selector = gsap.utils.selector(shellRef);
      const entranceTargets = selector(".top-bar, .page-tabs, .stage-toolbar");
      if (entranceTargets.length) {
        gsap.from(entranceTargets, {
          opacity: 0,
          y: 14,
          duration: motionDurations.panel,
          stagger: motionStagger.section,
          ease: motionEase.standard,
          clearProps: "all"
        });
      }
      const controlDrawer = selector(".left-control-drawer");
      if (controlDrawer.length) {
        gsap.from(controlDrawer, { opacity: 0, duration: motionDurations.panel, delay: 0.18, clearProps: "opacity" });
      }
      const dock = dockRef.current;
      if (dock) {
        // the dock keeps its own transform (drawer position), so fade opacity only
        gsap.from(dock, { opacity: 0, duration: motionDurations.panel, delay: 0.25, clearProps: "opacity" });
      }
    },
    { scope: shellRef }
  );

  // drawer slide; the closed position matches the CSS default transform
  useGSAP(
    () => {
      const dock = dockRef.current;
      if (!dock) return;
      const y = dockOpen ? 0 : dock.offsetHeight - DOCK_HANDLE_PX;
      if (reducedMotion) {
        gsap.set(dock, { y });
      } else {
        gsap.to(dock, { y, duration: motionDuration("drawer", reducedMotion), ease: motionEase.panel });
      }
    },
    { dependencies: [dockOpen, workspace, reducedMotion], scope: shellRef }
  );

  useEffect(() => {
    if (stream.status === "streaming") setDockOpen(true);
  }, [stream.status]);

  const loadDemoGraph = useCallback(() => {
    getDemoModel()
      .then((modelGraph) => {
      setGraph(modelGraph);
      setSelectedNode(firstDisplayNode(modelGraph));
      setInspectedNodeId(undefined);
      setGhostEdges([]);
      setSelectedGhostEdgeId(undefined);
      })
      .catch(() => setErrorMessage("Could not load the demo model graph. Is the backend running?"));
  }, []);

  useEffect(() => {
    const checkHealth = () => {
      getHealth()
        .then(() => setBackendStatus("ok"))
        .catch(() => setBackendStatus("offline"));
    };
    checkHealth();
    const timer = window.setInterval(checkHealth, HEALTH_POLL_MS);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    loadDemoGraph();
  }, [loadDemoGraph]);

  const refreshRuns = useCallback(() => {
    return listRuns()
      .then(setLiveRuns)
      .catch(() => setLiveRuns([]));
  }, []);

  useEffect(() => {
    if (backendStatus !== "ok") return;
    const pollRuns = () => refreshRuns();
    pollRuns();
    const timer = window.setInterval(pollRuns, RUNS_POLL_MS);
    return () => window.clearInterval(timer);
  }, [backendStatus, refreshRuns]);

  // a live run's registered graph replaces the displayed model graph
  useEffect(() => {
    if (stream.graph) {
      setGraph(stream.graph);
      setSelectedNode(firstDisplayNode(stream.graph));
      setGhostEdges([]);
      setSelectedGhostEdgeId(undefined);
    }
  }, [stream.graph]);

  useEffect(() => {
    if (!stream.runId) {
      setActiveRunContract(undefined);
      return;
    }
    let cancelled = false;
    getRunDetail(stream.runId)
      .then((detail) => {
        if (!cancelled) setActiveRunContract(runContractFromConfig(detail.config));
      })
      .catch(() => {
        if (!cancelled) setActiveRunContract(undefined);
      });
    return () => {
      cancelled = true;
    };
  }, [stream.runId]);

  const latestStep = stream.metrics.length ? stream.metrics[stream.metrics.length - 1].step : 0;
  const runBuckets = useMemo(() => splitRunBuckets(liveRuns), [liveRuns]);

  const handleResourceUpload = async (files: NamedSourceFile[]) => {
    stream.reset();
    setSelectedTimelineStep(undefined);
    setGhostEdges([]);
    setSelectedGhostEdgeId(undefined);
    setBusy("resource");
    setErrorMessage(undefined);
    const entryFile = preferredResourceEntry(files);
    if (!entryFile) {
      setErrorMessage("Upload a Python resource before training.");
      setBusy(undefined);
      return;
    }
    setPrediction(undefined);
    setForwardTarget(undefined);
    setPendingForwardRun(undefined);
    setCurrentRunKind(undefined);
    try {
      // trace the resource right away so the operator graph appears on import
      const preview = await previewResource(files, entryFile);
      setGraph(preview.graph);
      setSelectedNode(firstDisplayNode(preview.graph));
      setInspectedNodeId(undefined);
      setSourceRecipe({
        files,
        entryFile,
        summary: {
          name: preview.resource.name,
          fileCount: preview.files.length,
          task: preview.resource.task ?? undefined,
          datasetSpec: preview.resource.dataset_spec ?? null,
          outputSchema: preview.resource.output_schema ?? null,
          metricSchema: preview.resource.metric_schema ?? null,
          inputShape: preview.resource.input_shape ?? undefined,
          classes: preview.resource.classes ?? undefined,
          classNames: preview.resource.class_names ?? undefined,
          dataSource: preview.resource.data_source ?? undefined,
          samples: preview.samples
        }
      });
    } catch (error) {
      setSourceRecipe(undefined);
      setErrorMessage(error instanceof Error ? error.message : "Analyzing the resource failed.");
    } finally {
      setBusy(undefined);
    }
  };

  const handleRunForward = async () => {
    const target = resolveForwardTarget(forwardTarget);
    if (target.mode !== "run") {
      setErrorMessage("Run training before inference.");
      return;
    }
    setBusy("forward");
    setErrorMessage(undefined);
    try {
      const index = Math.floor(Math.random() * 20);
      const result = await runForward(target.runId, target.checkpointStep, index);
      applyPredictionResult(result);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Forward pass failed.");
    } finally {
      setBusy(undefined);
    }
  };

  const handleRunTraining = async () => {
    if (!sourceRecipe) {
      setErrorMessage("Upload a training resource before training.");
      return;
    }
    stream.reset();
    setSelectedTimelineStep(undefined);
    setGhostEdges([]);
    setSelectedGhostEdgeId(undefined);
    setBusy("train");
    setErrorMessage(undefined);
    try {
      const steps = Math.max(1, Math.min(500, Math.trunc(trainingSteps || DEFAULT_TRAINING_STEPS)));
      const stride = Math.max(1, Math.min(steps, Math.trunc(telemetryStride || DEFAULT_TELEMETRY_STRIDE)));
      const result = await trainResourceRun(sourceRecipe.files, sourceRecipe.entryFile, steps, stride);
      setGraph(result.graph);
      setSelectedNode(firstDisplayNode(result.graph));
      setInspectedNodeId(undefined);
      setPrediction(undefined);
      setForwardTarget({ runId: result.run_id });
      setPendingForwardRun(result.run_id);
      setCurrentRunKind("resource-training");
      selectWorkspace("train");
      stream.startStream(result.run_id);
      void refreshRuns();
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Training resource failed.");
    } finally {
      setBusy(undefined);
    }
  };

  const applyPredictionResult = (result: PredictionResponse) => {
    setPrediction(result);
    setDockOpen(true);
    setForwardTick((tick) => tick + 1);
    setGraph(result.graph);
    setSelectedNode(firstDisplayNode(result.graph));
    setInspectedNodeId(undefined);
    setGhostEdges([]);
    setSelectedGhostEdgeId(undefined);
    const lastNode = result.graph.nodes[result.graph.nodes.length - 1];
    stream.applyPrediction(result.layers, lastNode?.id);
  };

  useEffect(() => {
    if (!pendingForwardRun) return;
    if (stream.status === "error" || (stream.status === "complete" && stream.progress?.phase !== "completed")) {
      setPendingForwardRun(undefined);
      return;
    }
    if (stream.status !== "complete") return;
    let cancelled = false;
    runForward(pendingForwardRun)
      .then((result) => {
        if (!cancelled) applyPredictionResult(result);
      })
      .catch((error) => {
        if (!cancelled) setErrorMessage(error instanceof Error ? error.message : "Forward pass failed.");
      })
      .finally(() => {
        if (!cancelled) setPendingForwardRun(undefined);
      });
    return () => {
      cancelled = true;
    };
  }, [pendingForwardRun, stream.progress?.phase, stream.status]);

  const handleWatchRun = (runId: string) => {
    setErrorMessage(undefined);
    setForwardTarget({ runId });
    setCurrentRunKind("recorded-training");
    selectWorkspace("train");
    setSelectedTimelineStep(undefined);
    setGhostEdges([]);
    setSelectedGhostEdgeId(undefined);
    stream.startStream(runId);
  };

  const handleDeleteRun = async (runId: string) => {
    setErrorMessage(undefined);
    try {
      await deleteRun(runId);
      if (stream.runId === runId) stream.reset();
      if (detailRunId === runId) setDetailRunId(undefined);
      setLiveRuns((runs) => runs.filter((run) => run.run_id !== runId));
      if (forwardTarget?.runId === runId) {
        setForwardTarget(undefined);
        setPrediction(undefined);
      }
      void refreshRuns();
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Deleting the run failed.");
    }
  };

  const handleCancelRun = async () => {
    if (!stream.runId || stream.status !== "streaming") return;
    setCancellingRunId(stream.runId);
    setErrorMessage(undefined);
    try {
      await cancelRun(stream.runId);
      void refreshRuns();
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Cancelling the run failed.");
    } finally {
      setCancellingRunId(undefined);
    }
  };

  const handleReset = () => {
    stream.reset();
    setPrediction(undefined);
    setErrorMessage(undefined);
    setDetailRunId(undefined);
    setForwardTarget(undefined);
    setPendingForwardRun(undefined);
    setSourceRecipe(undefined);
    setCurrentRunKind(undefined);
    setInspectedNodeId(undefined);
    setSelectedTimelineStep(undefined);
    setGhostEdges([]);
    setSelectedGhostEdgeId(undefined);
    loadDemoGraph();
  };

  const predictionKind = prediction ? inferenceOutputKind(prediction) : "";
  const classificationPrediction = prediction ? classificationOutputFromPrediction(prediction) : undefined;
  const predictionSummary = prediction
    ? classificationPrediction
      ? `${displayClassName(classificationPrediction.prediction, classificationPrediction.classNames)} · ${sampleSourceLabel[prediction.sample_source]} · ${prediction.weights === "trained" ? "trained" : "random"}`
      : `${predictionKind} · ${sampleSourceLabel[prediction.sample_source]} · ${prediction.weights === "trained" ? "trained" : "random"}`
    : "";
  const trainingLoop = useMemo(
    () =>
      deriveTrainingLoopModel({
        hasResource: Boolean(sourceRecipe),
        hasGraph: graph.nodes.length > 0,
        hasPrediction: Boolean(prediction),
        metrics: stream.metrics,
        events: stream.events,
        trainingStageEvents: stream.trainingStages,
        progress: stream.progress
      }),
    [sourceRecipe, graph.nodes.length, prediction, stream.metrics, stream.events, stream.trainingStages, stream.progress]
  );
  const selectedLayerHistory = selectedNode ? stream.layerHistory[selectedNode.id] ?? [] : [];
  const timelineFrames = useMemo(
    () => deriveTimelineFrames(stream.metrics, stream.events, stream.layerHistory),
    [stream.metrics, stream.events, stream.layerHistory]
  );
  const selectedTimelineFrameStep = resolveTimelineStep(timelineFrames, selectedTimelineStep);
  const timelineLive =
    selectedTimelineStep == null || !timelineFrames.length || selectedTimelineFrameStep === timelineFrames[timelineFrames.length - 1].step;
  const replayLayerSnapshots = useMemo(
    () => layerSnapshotsAtStep(stream.layerSnapshots, stream.layerHistory, timelineLive ? undefined : selectedTimelineFrameStep),
    [stream.layerSnapshots, stream.layerHistory, selectedTimelineFrameStep, timelineLive]
  );
  const replayEvents = useMemo(
    () => eventsAtTimelineStep(stream.events, timelineLive ? undefined : selectedTimelineFrameStep),
    [stream.events, selectedTimelineFrameStep, timelineLive]
  );
  const causalFocus = useMemo(
    () =>
      deriveCausalFocus({
        step: selectedTimelineFrameStep,
        metrics: stream.metrics,
        events: stream.events,
        graph,
        layerSnapshots: replayLayerSnapshots
      }),
    [graph, replayLayerSnapshots, selectedTimelineFrameStep, stream.events, stream.metrics]
  );
  const selectedLayerEvents = selectedNode ? replayEvents.filter((event) => event.layer === selectedNode.id) : [];
  const replayPulseNodeId = timelineLive ? stream.pulsedNodeId : causalFocus.layerId ?? stream.pulsedNodeId;
  const inspectedNode = inspectedNodeId === selectedNode?.id ? selectedNode : undefined;
  const selectedGhostEdge = ghostEdges.find((edge) => edge.id === selectedGhostEdgeId);
  const metricTask = activeRunContract?.task ?? prediction?.task ?? sourceRecipe?.summary?.task;
  const metricSchema = activeRunContract?.metricSchema ?? prediction?.metric_schema ?? sourceRecipe?.summary?.metricSchema ?? undefined;

  const handleTimelineStepChange = (step: number) => {
    const latestFrameStep = timelineFrames[timelineFrames.length - 1]?.step;
    setSelectedTimelineStep(step === latestFrameStep ? undefined : step);
  };

  const handleSelectNode = (node: GraphNode) => {
    setSelectedNode(node);
    setInspectedNodeId(node.id);
  };

  const handleTrainingStageSelect = (stage: TrainingLoopStage) => {
    if (stage.evidence === "prepare") {
      selectWorkspace("prepare");
      return;
    }
    if (stage.evidence === "evaluate") {
      selectWorkspace("evaluate");
      return;
    }
    if (stage.evidence === "checkpoint") {
      if (stream.runId) {
        setDetailInitialTab("overview");
        setDetailOrigin(undefined);
        setDetailRunId(stream.runId);
      }
      return;
    }
    selectWorkspace("train");
    setDockOpen(stage.evidence === "telemetry" || stage.evidence === "diagnostics");
  };

  const renderControlRail = (contextView: "resource" | "train" | "run") => (
    <ControlRail
      contextView={contextView}
      initiallyOpen={contextView !== "run"}
      onResourceUpload={handleResourceUpload}
      onRunTraining={handleRunTraining}
      onRunForward={handleRunForward}
      onReset={handleReset}
      onWatchRun={handleWatchRun}
      onOpenDetail={(runId) => {
        setDetailInitialTab("overview");
        setDetailOrigin(undefined);
        setDetailRunId(runId);
      }}
      onViewAllRuns={() => {
        setHistoryMode("live");
        selectWorkspace("runs");
      }}
      trainAvailable={Boolean(sourceRecipe) && !(currentRunKind === "resource-training" && stream.status === "streaming")}
      loadedResource={sourceRecipe?.summary}
      trainingSteps={trainingSteps}
      onTrainingStepsChange={(steps) => setTrainingSteps(Math.max(1, Math.min(500, Math.trunc(steps || 1))))}
      telemetryStride={telemetryStride}
      onTelemetryStrideChange={(stride) => setTelemetryStride(Math.max(1, Math.min(500, Math.trunc(stride || 1))))}
      forwardTargetLabel={forwardTarget ? describeForwardTarget(forwardTarget) : "none"}
      currentRunKind={currentRunKind}
      hasPrediction={Boolean(prediction)}
      liveRuns={runBuckets.active}
      watchedRunId={stream.runId}
      busy={busy}
      errorMessage={errorMessage}
    />
  );

  return (
    <main className="app-shell" ref={shellRef}>
      <TopStatusBar
        backendStatus={backendStatus}
        runStatus={stream.status}
        step={stream.progress?.step ?? latestStep}
        device={stream.device}
        theme={theme}
        onToggleTheme={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
      />
      <nav className="page-tabs workspace-tabs" aria-label="PulseGraph workspaces">
        {workspaces.map((item) => (
          <button
            className={workspace === item.value ? "active" : ""}
            key={item.value}
            onClick={() => {
              if (item.value === "runs") setHistoryMode("completed");
              selectWorkspace(item.value);
            }}
            type="button"
          >
            {item.label}
            {item.value === "runs" ? <span>{runBuckets.history.length}</span> : null}
          </button>
        ))}
      </nav>

      {workspace === "prepare" || workspace === "train" ? (
        <div className={`stage workspace-stage workspace-${workspace}`}>
          {workspace === "train" ? (
            <TrainingLoopStrip model={trainingLoop} onStageSelect={handleTrainingStageSelect} />
          ) : null}
          <ModelGraphPanel
            graph={graph}
            selectedNodeId={selectedNode?.id}
            pulsedNodeId={replayPulseNodeId}
            probabilities={prediction?.probabilities}
            forwardTick={forwardTick}
            layerSnapshots={replayLayerSnapshots}
            ghostEdges={ghostEdges}
            selectedGhostEdgeId={selectedGhostEdgeId}
            onGhostEdgesChange={setGhostEdges}
            onGhostEdgeSelect={(edge) => {
              setSelectedGhostEdgeId(edge?.id);
              if (edge) setDockOpen(true);
            }}
            onSelect={handleSelectNode}
          />
          {inspectedNode && (
            <div className="graph-layer-detail-drawer">
              <LayerInspector
                node={inspectedNode}
                snapshot={replayLayerSnapshots[inspectedNode.id]}
                history={timelineLive ? selectedLayerHistory : selectedLayerHistory.filter((point) => point.step <= (selectedTimelineFrameStep ?? 0))}
                events={selectedLayerEvents}
                selectedStep={timelineLive ? undefined : selectedTimelineFrameStep}
                onClose={() => setInspectedNodeId(undefined)}
              />
            </div>
          )}
          {workspace === "train" ? <StageStats metrics={stream.metrics} task={metricTask} metricSchema={metricSchema} /> : null}
          {renderControlRail(workspace === "prepare" ? "resource" : "train")}
          {workspace === "train" ? (
          <section className={`bottom-dock train-bottom-dock ${dockOpen ? "open" : ""}`} ref={dockRef}>
            <header className="dock-handle">
              <button className="dock-toggle" type="button" onClick={() => setDockOpen((open) => !open)} aria-expanded={dockOpen}>
                <i className={`status-dot ${stream.status === "streaming" ? "streaming" : stream.status}`} />
                Telemetry
                <span className="dock-run">{stream.runId ?? ""}</span>
                {dockOpen ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
              </button>
            </header>
            <div className="dock-panels train-dock-panels">
              <TelemetryPanel
                points={stream.metrics}
                progress={stream.progress}
                runId={stream.runId}
                status={stream.status}
                theme={theme}
                runKind={currentRunKind}
                task={metricTask}
                metricSchema={metricSchema}
                selectedStep={timelineLive ? undefined : selectedTimelineFrameStep}
                cancelling={cancellingRunId === stream.runId}
                onCancel={stream.runId ? handleCancelRun : undefined}
                timeline={<TimelineScrubber
                  frames={timelineFrames}
                  selectedStep={selectedTimelineFrameStep}
                  live={timelineLive}
                  focus={causalFocus}
                  onStepChange={handleTimelineStepChange}
                  onLive={() => setSelectedTimelineStep(undefined)}
                  onJumpToStep={handleTimelineStepChange}
                />}
              />
              <DiagnosticsTray error={errorMessage} events={timelineLive ? stream.events : replayEvents}>
                {selectedGhostEdge && (
                    <div className={`composer-ghost-card ghost-${selectedGhostEdge.status}`}>
                      <div>
                        <span>Composer ghost edge</span>
                        <strong>
                          {selectedGhostEdge.sourcePort.nodeId}
                          {" -> "}
                          {selectedGhostEdge.targetPort.nodeId}
                        </strong>
                      </div>
                      <em>{selectedGhostEdge.status}</em>
                      {selectedGhostEdge.reasons.map((reason) => (
                        <p key={reason}>{reason}</p>
                      ))}
                      <button
                        type="button"
                        onClick={() => {
                          setGhostEdges((edges) => edges.filter((edge) => edge.id !== selectedGhostEdge.id));
                          setSelectedGhostEdgeId(undefined);
                        }}
                      >
                        Remove ghost edge
                      </button>
                    </div>
                )}
              </DiagnosticsTray>
            </div>
          </section>
          ) : null}
        </div>
      ) : workspace === "evaluate" ? (
        <section className="evaluate-workspace">
          <header className="evaluate-workspace-header">
            <div>
              <span>Evaluate</span>
              <h2>{sourceRecipe?.summary?.name ?? activeRunContract?.task ?? "Active run"}</h2>
            </div>
            <div className="evaluate-header-actions">
              <div className="evaluate-run-context">
                <span>{stream.runId ?? "No run selected"}</span>
                <em>{predictionSummary || "No inference result"}</em>
              </div>
              <button disabled={!forwardTarget || busy === "forward"} onClick={handleRunForward} type="button">
                <Zap size={14} /> Run Inference
              </button>
            </div>
          </header>
          <div className="evaluate-workspace-body">
            <section className="evaluate-output-surface">
              <div className="panel-heading">
                <h2>Inference Output</h2>
                <span>{predictionSummary}</span>
              </div>
              {prediction ? (
                <InferenceProbe prediction={prediction} theme={theme} />
              ) : (
                <div className="evaluate-empty-state">
                  <span>No inference result</span>
                  <em>{stream.runId ?? "No active run"}</em>
                </div>
              )}
            </section>
            <DiagnosticsTray error={errorMessage} events={timelineLive ? stream.events : replayEvents} />
          </div>
          {renderControlRail("run")}
        </section>
      ) : (
        <HistoryPage
          key={historyMode}
          runs={historyMode === "live" ? runBuckets.active : runBuckets.history}
          initialStatusFilter={historyMode === "live" ? "live" : "all"}
          statusFilters={historyMode === "live" ? ["live"] : undefined}
          title={historyMode === "live" ? "Live Runs" : "Run Library"}
          watchedRunId={stream.runId}
          onWatchRun={handleWatchRun}
          onOpenDetail={(runId, origin) => {
            setDetailInitialTab("overview");
            setDetailOrigin(origin);
            setDetailRunId(runId);
          }}
          onDeleteRun={handleDeleteRun}
        />
      )}

      {detailRunId && (
        <RunDetailPanel
          runId={detailRunId}
          initialTab={detailInitialTab}
          origin={detailOrigin}
          onClose={() => setDetailRunId(undefined)}
          onPrediction={(result) => {
            applyPredictionResult(result);
            setDetailRunId(undefined);
          }}
        />
      )}
    </main>
  );
}
