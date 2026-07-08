import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import type { GraphNode, ModelGraph, NamedSourceFile, PredictionResponse, RunSummary } from "./api/client";
import {
  deleteRun,
  getDemoModel,
  getHealth,
  listRuns,
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
import { useRunStream } from "./hooks/useRunStream";
import { useReducedMotion } from "./hooks/useReducedMotion";
import type { Theme } from "./lib/chartTheme";
import type { ForwardTarget } from "./lib/forwardTarget";
import { describeForwardTarget, resolveForwardTarget } from "./lib/forwardTarget";
import { firstDisplayNode } from "./lib/graphView";
import { splitRunBuckets } from "./lib/runViews";

gsap.registerPlugin(useGSAP);

const emptyGraph: ModelGraph = { nodes: [], edges: [] };
const HEALTH_POLL_MS = 8000;
const RUNS_POLL_MS = 5000;
const THEME_KEY = "pulsegraph-theme";

const sampleSourceLabel: Record<PredictionResponse["sample_source"], string> = {
  mnist: "MNIST",
  probe: "probe",
  synthetic: "synthetic"
};

function initialTheme(): Theme {
  const saved = window.localStorage.getItem(THEME_KEY);
  return saved === "light" ? "light" : "dark";
}

type SourceRecipe = {
  files: NamedSourceFile[];
  entryFile: string;
};

type CurrentRunKind = "resource-training" | "source-training" | "recorded-training";

export default function App() {
  const [backendStatus, setBackendStatus] = useState("checking");
  const [graph, setGraph] = useState<ModelGraph>(emptyGraph);
  const [selectedNode, setSelectedNode] = useState<GraphNode | undefined>();
  const [prediction, setPrediction] = useState<PredictionResponse | undefined>();
  const [liveRuns, setLiveRuns] = useState<RunSummary[]>([]);
  const [busy, setBusy] = useState<"resource" | "train" | "forward" | undefined>();
  const [errorMessage, setErrorMessage] = useState<string | undefined>();
  const [detailRunId, setDetailRunId] = useState<string | undefined>();
  const [detailInitialTab, setDetailInitialTab] = useState<"overview" | "source">("overview");
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [page, setPage] = useState<"monitor" | "history">("monitor");
  const [forwardTarget, setForwardTarget] = useState<ForwardTarget | undefined>();
  const [pendingForwardRun, setPendingForwardRun] = useState<string | undefined>();
  const [sourceRecipe, setSourceRecipe] = useState<SourceRecipe | undefined>();
  const [currentRunKind, setCurrentRunKind] = useState<CurrentRunKind | undefined>();
  const shellRef = useRef<HTMLElement | null>(null);
  const reducedMotion = useReducedMotion();
  const stream = useRunStream();

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useGSAP(
    () => {
      if (reducedMotion) return;
      gsap.from(".top-bar, .workspace > *, .bottom-dock > *", {
        opacity: 0,
        y: 14,
        duration: 0.5,
        stagger: 0.08,
        ease: "power2.out",
        clearProps: "all"
      });
    },
    { scope: shellRef }
  );

  const loadDemoGraph = useCallback(() => {
    getDemoModel()
      .then((modelGraph) => {
        setGraph(modelGraph);
        setSelectedNode(firstDisplayNode(modelGraph));
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
    }
  }, [stream.graph]);

  const latestStep = stream.metrics.length ? stream.metrics[stream.metrics.length - 1].step : 0;
  const runBuckets = useMemo(() => splitRunBuckets(liveRuns), [liveRuns]);

  const handleResourceUpload = (files: NamedSourceFile[]) => {
    stream.reset();
    setBusy("resource");
    setErrorMessage(undefined);
    const entryFile = files[0]?.path;
    if (!entryFile) {
      setErrorMessage("Upload a Python resource before training.");
      setBusy(undefined);
      return;
    }
    setSourceRecipe({ files, entryFile });
    setPrediction(undefined);
    setForwardTarget(undefined);
    setPendingForwardRun(undefined);
    setCurrentRunKind(undefined);
    setBusy(undefined);
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
    setBusy("train");
    setErrorMessage(undefined);
    try {
      const result = await trainResourceRun(sourceRecipe.files, sourceRecipe.entryFile);
      setGraph(result.graph);
      setSelectedNode(firstDisplayNode(result.graph));
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
    setGraph(result.graph);
    setSelectedNode(firstDisplayNode(result.graph));
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
    loadDemoGraph();
  };

  const predictionSummary = prediction
    ? `${prediction.prediction} · ${sampleSourceLabel[prediction.sample_source]} · ${prediction.weights === "trained" ? "trained" : "random"}`
    : "";

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
        <div className="workspace">
          <ControlRail
            onResourceUpload={handleResourceUpload}
            onRunTraining={handleRunTraining}
            onRunForward={handleRunForward}
            onReset={handleReset}
            onWatchRun={handleWatchRun}
            onOpenDetail={(runId) => {
              setDetailInitialTab("overview");
              setDetailRunId(runId);
            }}
            trainAvailable={Boolean(sourceRecipe)}
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
          <ModelGraphPanel
            graph={graph}
            selectedNodeId={selectedNode?.id}
            pulsedNodeId={stream.pulsedNodeId}
            probabilities={prediction?.probabilities}
            onSelect={setSelectedNode}
          />
        </div>
      ) : (
        <HistoryPage
          runs={runBuckets.history}
          watchedRunId={stream.runId}
          onWatchRun={handleWatchRun}
          onOpenDetail={(runId) => {
            setDetailInitialTab("overview");
            setDetailRunId(runId);
          }}
          onDeleteRun={handleDeleteRun}
        />
      )}

      <section className={`bottom-dock ${page === "history" ? "history-dock" : ""}`}>
        <div className="metric-panel">
          <div className="panel-heading">
            <h2>Training Telemetry</h2>
            <span>{stream.runId ? stream.runId : ""}</span>
          </div>
          <MetricChart points={stream.metrics} status={stream.status} theme={theme} runKind={currentRunKind} />
        </div>
        <div className="prediction-panel">
          <div className="panel-heading">
            <h2>Recognition Result</h2>
            <span>{predictionSummary}</span>
          </div>
          <InferenceProbe prediction={prediction} theme={theme} />
        </div>
        <div className="event-panel">
          <div className="panel-heading">
            <h2>Runtime Events</h2>
            <span>{stream.events.length}</span>
          </div>
          <div className="event-list">
            {errorMessage && <div className="event warning">{errorMessage}</div>}
            {stream.events.length === 0 && !errorMessage && (
              <p className="empty-hint">No events</p>
            )}
            {stream.events.map((event) => (
              <div className={`event ${event.type}`} key={event.event_id}>
                <strong>{event.type}</strong> step {event.step} {event.layer ? `· ${event.layer}` : ""}
              </div>
            ))}
          </div>
        </div>
      </section>

      {detailRunId && (
        <RunDetailPanel
          runId={detailRunId}
          initialTab={detailInitialTab}
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
