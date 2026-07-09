import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import type { GraphNode, ImageSample, ModelGraph, NamedSourceFile, PredictionResponse, RunSummary } from "./api/client";
import {
  deleteRun,
  getDemoModel,
  getHealth,
  listRuns,
  previewResource,
  runForward,
  trainResourceRun
} from "./api/client";
import { TopStatusBar } from "./components/TopStatusBar";
import { ControlRail } from "./components/ControlRail";
import { ModelGraphPanel } from "./components/ModelGraphPanel";
import { MetricChart } from "./components/Charts";
import { InferenceProbe } from "./components/InferenceProbe";
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
import { displayClassName, inferenceOutputKind } from "./lib/inferenceView";
import { configureMotionDefaults, motionDuration, motionDurations, motionEase, motionStagger } from "./lib/motion";
import { splitRunBuckets } from "./lib/runViews";
import {
  deriveCausalFocus,
  deriveTimelineFrames,
  eventsAtTimelineStep,
  layerSnapshotsAtStep,
  resolveTimelineStep
} from "./lib/timeline";
import { deriveTrainingLoopStages } from "./lib/trainingLoop";

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

export default function App() {
  const [backendStatus, setBackendStatus] = useState("checking");
  const [graph, setGraph] = useState<ModelGraph>(emptyGraph);
  const [selectedNode, setSelectedNode] = useState<GraphNode | undefined>();
  const [inspectedNodeId, setInspectedNodeId] = useState<string | undefined>();
  const [prediction, setPrediction] = useState<PredictionResponse | undefined>();
  const [liveRuns, setLiveRuns] = useState<RunSummary[]>([]);
  const [busy, setBusy] = useState<"resource" | "train" | "forward" | undefined>();
  const [errorMessage, setErrorMessage] = useState<string | undefined>();
  const [detailRunId, setDetailRunId] = useState<string | undefined>();
  const [detailInitialTab, setDetailInitialTab] = useState<"overview" | "source">("overview");
  const [detailOrigin, setDetailOrigin] = useState<DOMRect | undefined>();
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [page, setPage] = useState<"monitor" | "history">("monitor");
  const [forwardTarget, setForwardTarget] = useState<ForwardTarget | undefined>();
  const [pendingForwardRun, setPendingForwardRun] = useState<string | undefined>();
  const [sourceRecipe, setSourceRecipe] = useState<SourceRecipe | undefined>();
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
    { dependencies: [dockOpen, page, reducedMotion], scope: shellRef }
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

  const latestStep = stream.metrics.length ? stream.metrics[stream.metrics.length - 1].step : 0;
  const runBuckets = useMemo(() => splitRunBuckets(liveRuns), [liveRuns]);

  const handleResourceUpload = async (files: NamedSourceFile[]) => {
    stream.reset();
    setSelectedTimelineStep(undefined);
    setGhostEdges([]);
    setSelectedGhostEdgeId(undefined);
    setBusy("resource");
    setErrorMessage(undefined);
    const entryFile = files[0]?.path;
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
      setPage("monitor");
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
    if (stream.status !== "complete" || !pendingForwardRun) return;
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
  }, [pendingForwardRun, stream.status]);

  const handleWatchRun = (runId: string) => {
    setErrorMessage(undefined);
    setForwardTarget({ runId });
    setCurrentRunKind("recorded-training");
    setPage("monitor");
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
  const predictionSummary = prediction
    ? predictionKind === "classification"
      ? `${displayClassName(prediction.prediction, prediction.class_names)} · ${sampleSourceLabel[prediction.sample_source]} · ${prediction.weights === "trained" ? "trained" : "random"}`
      : `${predictionKind} · ${sampleSourceLabel[prediction.sample_source]} · ${prediction.weights === "trained" ? "trained" : "random"}`
    : "";
  const loopStages = useMemo(
    () =>
      deriveTrainingLoopStages({
        hasResource: Boolean(sourceRecipe),
        hasGraph: graph.nodes.length > 0,
        hasPrediction: Boolean(prediction),
        metrics: stream.metrics,
        events: stream.events
      }),
    [sourceRecipe, graph.nodes.length, prediction, stream.metrics, stream.events]
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

  const handleTimelineStepChange = (step: number) => {
    const latestFrameStep = timelineFrames[timelineFrames.length - 1]?.step;
    setSelectedTimelineStep(step === latestFrameStep ? undefined : step);
  };

  const handleSelectNode = (node: GraphNode) => {
    setSelectedNode(node);
    setInspectedNodeId(node.id);
  };

  return (
    <main className="app-shell" ref={shellRef}>
      <TopStatusBar
        backendStatus={backendStatus}
        runStatus={stream.status}
        step={latestStep}
        device={stream.device}
        theme={theme}
        onToggleTheme={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
      />
      <nav className="page-tabs" aria-label="PulseGraph views">
        <button className={page === "monitor" ? "active" : ""} onClick={() => setPage("monitor")} type="button">
          Monitor
        </button>
        <button className={page === "history" ? "active" : ""} onClick={() => setPage("history")} type="button">
          History <span>{runBuckets.history.length}</span>
        </button>
      </nav>

      {page === "monitor" ? (
        <div className="stage">
          <TrainingLoopStrip stages={loopStages} />
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
          <StageStats metrics={stream.metrics} />
          <ControlRail
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
            trainAvailable={Boolean(sourceRecipe)}
            loadedResource={sourceRecipe?.summary}
            trainingSteps={trainingSteps}
            onTrainingStepsChange={(steps) => setTrainingSteps(Math.max(1, Math.min(500, Math.trunc(steps || 1))))}
            telemetryStride={telemetryStride}
            onTelemetryStrideChange={(stride) => setTelemetryStride(Math.max(1, Math.min(500, Math.trunc(stride || 1))))}
            forwardTargetLabel={forwardTarget ? describeForwardTarget(forwardTarget) : "none"}
            currentRunKind={currentRunKind}
            metricCount={stream.metrics.length}
            eventCount={stream.events.length}
            hasPrediction={Boolean(prediction)}
            liveRuns={runBuckets.active}
            watchedRunId={stream.runId}
            busy={busy}
            errorMessage={errorMessage}
          />
          <section className={`bottom-dock ${dockOpen ? "open" : ""}`} ref={dockRef}>
            <button
              className="dock-handle"
              type="button"
              onClick={() => setDockOpen((open) => !open)}
              aria-expanded={dockOpen}
            >
              <i className={`status-dot ${stream.status === "streaming" ? "streaming" : "idle"}`} />
              Telemetry
              <span className="dock-run">{stream.runId ?? ""}</span>
              {dockOpen ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
            </button>
            <div className="dock-panels">
              <div className="metric-panel">
                <div className="panel-heading">
                  <h2>Training Telemetry</h2>
                  <span>{stream.runId ? stream.runId : ""}</span>
                </div>
                <MetricChart
                  points={stream.metrics}
                  status={stream.status}
                  theme={theme}
                  runKind={currentRunKind}
                  selectedStep={timelineLive ? undefined : selectedTimelineFrameStep}
                />
                <TimelineScrubber
                  frames={timelineFrames}
                  selectedStep={selectedTimelineFrameStep}
                  live={timelineLive}
                  focus={causalFocus}
                  onStepChange={handleTimelineStepChange}
                  onLive={() => setSelectedTimelineStep(undefined)}
                  onJumpToStep={handleTimelineStepChange}
                />
              </div>
              <div className="prediction-panel">
                <div className="panel-heading">
                  <h2>Inference Output</h2>
                  <span>{predictionSummary}</span>
                </div>
                <InferenceProbe prediction={prediction} theme={theme} />
              </div>
              <div className="event-panel">
                <div className="panel-heading">
                  <h2>Runtime Events</h2>
                  <span>{timelineLive ? stream.events.length : replayEvents.length}</span>
                </div>
                <div className="event-list">
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
                  {errorMessage && (
                    <div className="event warning">
                      <i className="event-dot" />
                      <span className="event-layer">{errorMessage}</span>
                    </div>
                  )}
                  {replayEvents.length === 0 && !errorMessage && (
                    <p className="empty-hint">No events</p>
                  )}
                  {replayEvents.map((event) => (
                    <div className={`event ${event.type}`} key={event.event_id}>
                      <i className="event-dot" />
                      <span className="event-type">{event.type}</span>
                      {event.layer && <span className="event-layer">{event.layer}</span>}
                      <span className="event-step">step {event.step}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>
        </div>
      ) : (
        <HistoryPage
          runs={runBuckets.history}
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
