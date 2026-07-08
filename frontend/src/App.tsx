import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import type { GraphNode, InspectionResponse, ModelGraph, NamedSourceFile, PredictionResponse, RunSummary } from "./api/client";
import {
  analyzeSourceCandidates,
  getDemoForward,
  getDemoModel,
  getHealth,
  importArtifact,
  importSourceRun,
  inspectFile,
  listRuns,
  runForward,
  trainSourceRun
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
  entryClass: string;
};

type CurrentRunKind = "source-import" | "source-training" | "recorded-training";

export default function App() {
  const [backendStatus, setBackendStatus] = useState("checking");
  const [graph, setGraph] = useState<ModelGraph>(emptyGraph);
  const [selectedNode, setSelectedNode] = useState<GraphNode | undefined>();
  const [inspection, setInspection] = useState<InspectionResponse | undefined>();
  const [prediction, setPrediction] = useState<PredictionResponse | undefined>();
  const [liveRuns, setLiveRuns] = useState<RunSummary[]>([]);
  const [busy, setBusy] = useState<"inspect" | "source" | "train" | "forward" | undefined>();
  const [errorMessage, setErrorMessage] = useState<string | undefined>();
  const [detailRunId, setDetailRunId] = useState<string | undefined>();
  const [detailInitialTab, setDetailInitialTab] = useState<"overview" | "source">("overview");
  const [importCandidate, setImportCandidate] = useState<File | undefined>();
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

  useEffect(() => {
    if (backendStatus !== "ok") return;
    const pollRuns = () => {
      listRuns()
        .then(setLiveRuns)
        .catch(() => setLiveRuns([]));
    };
    pollRuns();
    const timer = window.setInterval(pollRuns, RUNS_POLL_MS);
    return () => window.clearInterval(timer);
  }, [backendStatus]);

  // a live run's registered graph replaces the displayed model graph
  useEffect(() => {
    if (stream.graph) {
      setGraph(stream.graph);
      setSelectedNode(firstDisplayNode(stream.graph));
    }
  }, [stream.graph]);

  const latestStep = stream.metrics.length ? stream.metrics[stream.metrics.length - 1].step : 0;
  const runBuckets = useMemo(() => splitRunBuckets(liveRuns), [liveRuns]);

  const handleInspect = async (file: File) => {
    stream.reset();
    setBusy("inspect");
    setErrorMessage(undefined);
    try {
      const result = await inspectFile(file);
      setInspection(result);
      setGraph(result.graph);
      setSelectedNode(firstDisplayNode(result.graph));
      setPrediction(undefined);
      setImportCandidate(undefined);
      if (result.matched_run_id) {
        setForwardTarget({ runId: result.matched_run_id });
        setCurrentRunKind("recorded-training");
        setDetailInitialTab("overview");
        setDetailRunId(result.matched_run_id);
      } else if (result.weights_fingerprint) {
        setForwardTarget(undefined);
        setImportCandidate(file);
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Inspection failed.");
    } finally {
      setBusy(undefined);
    }
  };

  const handleRunForward = async () => {
    setBusy("forward");
    setErrorMessage(undefined);
    try {
      const index = Math.floor(Math.random() * 20);
      const target = resolveForwardTarget(forwardTarget);
      const result =
        target.mode === "run"
          ? await runForward(target.runId, target.checkpointStep, index)
          : await getDemoForward(index);
      applyPredictionResult(result);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Forward pass failed.");
    } finally {
      setBusy(undefined);
    }
  };

  const handleSourceImport = async (files: NamedSourceFile[]) => {
    stream.reset();
    setBusy("source");
    setErrorMessage(undefined);
    try {
      const analysis = await analyzeSourceCandidates(files);
      const selected = analysis.candidates[0];
      if (!selected) throw new Error("No nn.Module subclass was found in the uploaded files.");
      setSourceRecipe({ files, entryFile: selected.file, entryClass: selected.class_name });
      const result = await importSourceRun(files, selected.file, selected.class_name);
      setInspection(undefined);
      setImportCandidate(undefined);
      setGraph(result.graph);
      setSelectedNode(firstDisplayNode(result.graph));
      setForwardTarget({ runId: result.run_id, checkpointStep: result.checkpoint?.step ?? 0 });
      setCurrentRunKind("source-import");
      setPage("monitor");
      stream.startStream(result.run_id);
      applyPredictionResult(await runForward(result.run_id, result.checkpoint?.step ?? 0));
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Creating a run from source failed.");
    } finally {
      setBusy(undefined);
    }
  };

  const handleSourceTrain = async () => {
    if (!sourceRecipe) {
      setErrorMessage("Upload Python source before training.");
      return;
    }
    stream.reset();
    setBusy("train");
    setErrorMessage(undefined);
    try {
      const result = await trainSourceRun(sourceRecipe.files, sourceRecipe.entryFile, sourceRecipe.entryClass);
      setInspection(undefined);
      setImportCandidate(undefined);
      setGraph(result.graph);
      setSelectedNode(firstDisplayNode(result.graph));
      setPrediction(undefined);
      setForwardTarget({ runId: result.run_id });
      setPendingForwardRun(result.run_id);
      setCurrentRunKind("source-training");
      setPage("monitor");
      stream.startStream(result.run_id);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Training source failed.");
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

  const handleStartStream = () => {
    setErrorMessage(undefined);
    const target = resolveForwardTarget(forwardTarget);
    stream.startStream(target.mode === "run" ? target.runId : undefined);
  };

  const handleWatchRun = (runId: string) => {
    setErrorMessage(undefined);
    setForwardTarget({ runId });
    setCurrentRunKind("recorded-training");
    setPage("monitor");
    stream.startStream(runId);
  };

  const handleImportForAttach = async () => {
    if (!importCandidate) return;
    setErrorMessage(undefined);
    try {
      const result = await importArtifact(importCandidate);
      setImportCandidate(undefined);
      setDetailInitialTab("source");
      setDetailRunId(result.run_id);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Importing the artifact failed.");
    }
  };

  const handleReset = () => {
    stream.reset();
    setPrediction(undefined);
    setInspection(undefined);
    setErrorMessage(undefined);
    setDetailRunId(undefined);
    setImportCandidate(undefined);
    setForwardTarget(undefined);
    setPendingForwardRun(undefined);
    setSourceRecipe(undefined);
    setCurrentRunKind(undefined);
    loadDemoGraph();
  };

  const warnings = inspection?.warnings ?? [];
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
            onInspect={handleInspect}
            onSourceImport={handleSourceImport}
            onTrainSource={handleSourceTrain}
            onRunForward={handleRunForward}
            onStartStream={handleStartStream}
            onReset={handleReset}
            onWatchRun={handleWatchRun}
            onOpenDetail={(runId) => {
              setDetailInitialTab("overview");
              setDetailRunId(runId);
            }}
            onImportAttach={handleImportForAttach}
            importAvailable={Boolean(importCandidate)}
            trainAvailable={Boolean(sourceRecipe)}
            forwardTargetLabel={describeForwardTarget(forwardTarget)}
            currentRunKind={currentRunKind}
            metricCount={stream.metrics.length}
            eventCount={stream.events.length}
            hasPrediction={Boolean(prediction)}
            liveRuns={runBuckets.active}
            watchedRunId={stream.runId}
            busy={busy}
            streaming={stream.status === "streaming"}
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
            {warnings.map((warning) => (
              <div className="event warning" key={warning}>{warning}</div>
            ))}
            {stream.events.length === 0 && !errorMessage && warnings.length === 0 && (
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
