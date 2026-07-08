import { useCallback, useEffect, useMemo, useState } from "react";
import type { GraphNode, InspectionResponse, ModelGraph, PredictionResponse, RunSummary, TensorSummary } from "./api/client";
import { getDemoForward, getDemoModel, getHealth, inspectFile, listRuns } from "./api/client";
import { TopStatusBar } from "./components/TopStatusBar";
import { ControlRail } from "./components/ControlRail";
import { ModelGraphPanel } from "./components/ModelGraphPanel";
import { LayerInspector } from "./components/LayerInspector";
import { MetricChart, ProbabilityChart } from "./components/Charts";
import { DigitPreview } from "./components/DigitPreview";
import { useRunStream } from "./hooks/useRunStream";

const emptyGraph: ModelGraph = { nodes: [], edges: [] };
const HEALTH_POLL_MS = 8000;
const RUNS_POLL_MS = 5000;

export default function App() {
  const [backendStatus, setBackendStatus] = useState("checking");
  const [graph, setGraph] = useState<ModelGraph>(emptyGraph);
  const [selectedNode, setSelectedNode] = useState<GraphNode | undefined>();
  const [inspection, setInspection] = useState<InspectionResponse | undefined>();
  const [prediction, setPrediction] = useState<PredictionResponse | undefined>();
  const [liveRuns, setLiveRuns] = useState<RunSummary[]>([]);
  const [busy, setBusy] = useState<"inspect" | "forward" | undefined>();
  const [errorMessage, setErrorMessage] = useState<string | undefined>();
  const stream = useRunStream();

  const loadDemoGraph = useCallback(() => {
    getDemoModel()
      .then((modelGraph) => {
        setGraph(modelGraph);
        setSelectedNode(modelGraph.nodes[0]);
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

  const selectedLayer = selectedNode ? stream.layerSnapshots[selectedNode.id] : undefined;
  const tensors: TensorSummary[] = inspection?.tensors ?? [];
  const latestStep = stream.metrics.length ? stream.metrics[stream.metrics.length - 1].step : 0;

  const handleInspect = async (file: File) => {
    stream.reset();
    setBusy("inspect");
    setErrorMessage(undefined);
    try {
      const result = await inspectFile(file);
      setInspection(result);
      setGraph(result.graph);
      setSelectedNode(result.graph.nodes[0]);
      setPrediction(undefined);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Inspection failed.");
    } finally {
      setBusy(undefined);
    }
  };

  const handleDemoForward = async () => {
    setBusy("forward");
    setErrorMessage(undefined);
    try {
      const index = Math.floor(Math.random() * 20);
      const result = await getDemoForward(index);
      setPrediction(result);
      setGraph(result.graph);
      setSelectedNode(result.graph.nodes[0]);
      const lastNode = result.graph.nodes[result.graph.nodes.length - 1];
      stream.applyPrediction(result.layers, lastNode?.id);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Forward pass failed.");
    } finally {
      setBusy(undefined);
    }
  };

  const handleStartStream = () => {
    setErrorMessage(undefined);
    stream.startStream();
  };

  const handleWatchRun = (runId: string) => {
    setErrorMessage(undefined);
    stream.startStream(runId);
  };

  const handleReset = () => {
    stream.reset();
    setPrediction(undefined);
    setInspection(undefined);
    setErrorMessage(undefined);
    loadDemoGraph();
  };

  const warnings = inspection?.warnings ?? [];
  const selectedMetadata = useMemo(
    () => (selectedNode && Object.keys(selectedNode.metadata).length ? JSON.stringify(selectedNode.metadata, null, 2) : ""),
    [selectedNode]
  );

  const predictionSummary = prediction
    ? `label ${prediction.label} · predicted ${prediction.prediction} · ${prediction.sample_source === "mnist" ? "MNIST test set" : "synthetic sample"} · ${prediction.weights === "trained" ? "trained weights" : "random weights"}`
    : "run forward to inspect probabilities";

  return (
    <main className="app-shell">
      <TopStatusBar backendStatus={backendStatus} runStatus={stream.status} step={latestStep} device={stream.device} />
      <div className="workspace">
        <ControlRail
          onInspect={handleInspect}
          onDemoForward={handleDemoForward}
          onStartStream={handleStartStream}
          onReset={handleReset}
          onWatchRun={handleWatchRun}
          liveRuns={liveRuns}
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
        <LayerInspector node={selectedNode} tensors={tensors} layer={selectedLayer} inspection={inspection} />
      </div>

      <section className="bottom-dock">
        <div className="metric-panel">
          <div className="panel-heading">
            <h2>Training Telemetry</h2>
            <span>{stream.runId ? `run ${stream.runId}` : "loss, accuracy, step time"}</span>
          </div>
          <MetricChart points={stream.metrics} status={stream.status} />
        </div>
        <div className="prediction-panel">
          <div className="panel-heading">
            <h2>Inference Probe</h2>
            <span>{predictionSummary}</span>
          </div>
          <div className="inference-body">
            <DigitPreview pixels={prediction?.image_pixels} label={prediction?.label} prediction={prediction?.prediction} />
            <ProbabilityChart probabilities={prediction?.probabilities ?? Array(10).fill(0)} />
          </div>
        </div>
        <div className="event-panel">
          <div className="panel-heading">
            <h2>Runtime Events</h2>
            <span>{stream.events.length} recent events</span>
          </div>
          <div className="event-list">
            {errorMessage && <div className="event warning">{errorMessage}</div>}
            {warnings.map((warning) => (
              <div className="event warning" key={warning}>{warning}</div>
            ))}
            {selectedMetadata && (
              <div className="event metadata">
                <pre>{selectedMetadata}</pre>
              </div>
            )}
            {stream.events.length === 0 && !errorMessage && warnings.length === 0 && (
              <p className="empty-hint">Start a demo stream or watch a live training run to see events here.</p>
            )}
            {stream.events.map((event) => (
              <div className={`event ${event.type}`} key={event.event_id}>
                <strong>{event.type}</strong> step {event.step} {event.layer ? `· ${event.layer}` : ""}
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
