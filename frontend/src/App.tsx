import { useEffect, useMemo, useRef, useState } from "react";
import type { GraphNode, InspectionResponse, LayerSnapshot, ModelGraph, PredictionResponse, RunEvent, TensorSummary } from "./api/client";
import { getDemoForward, getDemoModel, getHealth, inspectFile, openDemoStream } from "./api/client";
import { TopStatusBar } from "./components/TopStatusBar";
import { ControlRail } from "./components/ControlRail";
import { ModelGraphPanel } from "./components/ModelGraphPanel";
import { LayerInspector } from "./components/LayerInspector";
import { MetricChart, ProbabilityChart, type MetricPoint } from "./components/Charts";

const emptyGraph: ModelGraph = { nodes: [], edges: [] };

export default function App() {
  const [backendStatus, setBackendStatus] = useState("checking");
  const [graph, setGraph] = useState<ModelGraph>(emptyGraph);
  const [selectedNode, setSelectedNode] = useState<GraphNode | undefined>();
  const [inspection, setInspection] = useState<InspectionResponse | undefined>();
  const [prediction, setPrediction] = useState<PredictionResponse | undefined>();
  const [layerSnapshots, setLayerSnapshots] = useState<Record<string, LayerSnapshot>>({});
  const [metrics, setMetrics] = useState<MetricPoint[]>([]);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [runStatus, setRunStatus] = useState("idle");
  const [pulsedNodeId, setPulsedNodeId] = useState<string | undefined>();
  const [device, setDevice] = useState("unknown");
  const streamRef = useRef<EventSource | null>(null);

  useEffect(() => {
    getHealth()
      .then(() => setBackendStatus("ok"))
      .catch(() => setBackendStatus("offline"));
    getDemoModel().then((modelGraph) => {
      setGraph(modelGraph);
      setSelectedNode(modelGraph.nodes[0]);
    });

    return () => streamRef.current?.close();
  }, []);

  const selectedLayer = selectedNode ? layerSnapshots[selectedNode.id] : undefined;
  const tensors: TensorSummary[] = inspection?.tensors ?? [];
  const latestStep = metrics.length ? metrics[metrics.length - 1].step : 0;

  const handleInspect = async (file: File) => {
    streamRef.current?.close();
    streamRef.current = null;
    const result = await inspectFile(file);
    setInspection(result);
    setGraph(result.graph);
    setSelectedNode(result.graph.nodes[0]);
    setPrediction(undefined);
    setLayerSnapshots({});
    setPulsedNodeId(undefined);
    setMetrics([]);
    setEvents([]);
    setRunStatus("inspected");
  };

  const handleDemoForward = async () => {
    const index = Math.floor(Math.random() * 20);
    const result = await getDemoForward(index);
    setPrediction(result);
    setGraph(result.graph);
    setSelectedNode(result.graph.nodes[0]);
    const nextSnapshots: Record<string, LayerSnapshot> = {};
    for (const layer of result.layers) nextSnapshots[layer.layer_id] = layer;
    setLayerSnapshots(nextSnapshots);
    setPulsedNodeId(result.graph.nodes.length ? result.graph.nodes[result.graph.nodes.length - 1].id : undefined);
  };

  const handleStartStream = () => {
    streamRef.current?.close();
    setRunStatus("streaming");
    setMetrics([]);
    setEvents([]);
    const source = openDemoStream((event) => {
      setEvents((previous) => [event, ...previous].slice(0, 40));
      if (event.type === "metric") {
        setMetrics((previous) => [
          ...previous,
          {
            step: event.step,
            loss: Number(event.payload.loss),
            accuracy: Number(event.payload.accuracy)
          }
        ]);
      }
      if (event.type === "infra") {
        setDevice(String(event.payload.device ?? "unknown"));
        setMetrics((previous) => {
          const next = [...previous];
          const last = next[next.length - 1];
          if (last?.step === event.step) {
            last.stepTimeMs = Number(event.payload.step_time_ms);
            last.memoryPeakMb = Number(event.payload.memory_peak_mb);
          }
          return next;
        });
      }
      if (event.type === "layer_snapshot" && event.layer) {
        setLayerSnapshots((previous) => ({
          ...previous,
          [event.layer as string]: {
            layer_id: event.layer as string,
            activation_mean: Number(event.payload.activation_mean),
            activation_sparsity: Number(event.payload.activation_sparsity),
            gradient_norm: Number(event.payload.gradient_norm),
            weight_std: Number(event.payload.weight_std)
          }
        }));
        setPulsedNodeId(event.layer);
      }
      if (event.type === "animation") {
        const path = event.payload.path as string[] | undefined;
        setPulsedNodeId(path?.[event.step % (path.length || 1)] ?? "linear1");
      }
      if (event.type === "run_complete") {
        setRunStatus("complete");
        source.close();
        streamRef.current = null;
      }
    });
    streamRef.current = source;
  };

  const handleReset = () => {
    streamRef.current?.close();
    streamRef.current = null;
    setMetrics([]);
    setEvents([]);
    setPrediction(undefined);
    setInspection(undefined);
    setLayerSnapshots({});
    setRunStatus("idle");
    setPulsedNodeId(undefined);
    getDemoModel().then((modelGraph) => {
      setGraph(modelGraph);
      setSelectedNode(modelGraph.nodes[0]);
    });
  };

  const warnings = inspection?.warnings ?? [];
  const selectedMetadata = useMemo(() => selectedNode ? JSON.stringify(selectedNode.metadata, null, 2) : "", [selectedNode]);

  return (
    <main className="app-shell">
      <TopStatusBar backendStatus={backendStatus} runStatus={runStatus} step={latestStep} device={device} />
      <div className="workspace">
        <ControlRail onInspect={handleInspect} onDemoForward={handleDemoForward} onStartStream={handleStartStream} onReset={handleReset} />
        <ModelGraphPanel
          graph={graph}
          selectedNodeId={selectedNode?.id}
          pulsedNodeId={pulsedNodeId}
          probabilities={prediction?.probabilities}
          onSelect={setSelectedNode}
        />
        <LayerInspector node={selectedNode} tensors={tensors} layer={selectedLayer} inspection={inspection} />
      </div>

      <section className="bottom-dock">
        <div className="metric-panel">
          <div className="panel-heading">
            <h2>Training Pulse</h2>
            <span>loss, accuracy, step time</span>
          </div>
          <MetricChart points={metrics} />
        </div>
        <div className="prediction-panel">
          <div className="panel-heading">
            <h2>Trusted Forward</h2>
            <span>{prediction ? `label ${prediction.label} · predicted ${prediction.prediction}` : "run forward to inspect probabilities"}</span>
          </div>
          <ProbabilityChart probabilities={prediction?.probabilities ?? Array(10).fill(0)} />
        </div>
        <div className="event-panel">
          <div className="panel-heading">
            <h2>Event Stream</h2>
            <span>{events.length} recent events</span>
          </div>
          <div className="event-list">
            {warnings.map((warning) => <div className="event warning" key={warning}>{warning}</div>)}
            {selectedMetadata && <div className="event metadata"><pre>{selectedMetadata}</pre></div>}
            {events.map((event) => (
              <div className={`event ${event.type}`} key={`${event.type}-${event.step}-${event.layer ?? ""}`}>
                <strong>{event.type}</strong> step {event.step} {event.layer ? `· ${event.layer}` : ""}
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
