import { memo, useEffect, useMemo, useRef, useState } from "react";
import { Handle, Position, ReactFlow, Background, Controls, MiniMap, type Edge, type Node, type NodeProps } from "@xyflow/react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import type { GraphNode, ModelGraph } from "../api/client";
import { NeuralNetworkView } from "./NeuralNetworkView";

gsap.registerPlugin(useGSAP);

function shapeText(shape?: number[] | null) {
  return shape && shape.length ? shape.join(" x ") : "unknown";
}

const PulseNode = memo(({ data, selected }: NodeProps<Node<GraphNode>>) => {
  return (
    <div className={`model-node ${selected ? "selected" : ""}`} data-layer-id={data.id}>
      <Handle type="target" position={Position.Left} />
      <div className="node-kind">{data.kind}</div>
      <div className="node-label">{data.label}</div>
      <div className="node-meta">out {shapeText(data.output_shape)}</div>
      <div className="node-meta">{data.param_count.toLocaleString()} params</div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
});

const nodeTypes = { pulse: PulseNode };

type Props = {
  graph: ModelGraph;
  selectedNodeId?: string;
  pulsedNodeId?: string;
  probabilities?: number[];
  onSelect: (node: GraphNode) => void;
};

export function ModelGraphPanel({ graph, selectedNodeId, pulsedNodeId, probabilities, onSelect }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [viewMode, setViewMode] = useState<"ops" | "neurons">("ops");

  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReducedMotion(query.matches);
    const onChange = () => setReducedMotion(query.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);
  const nodes = useMemo<Node<GraphNode>[]>(() => {
    return graph.nodes.map((node, index) => ({
      id: node.id,
      type: "pulse",
      position: { x: 80 + index * 230, y: index % 2 ? 190 : 70 },
      data: node,
      selected: node.id === selectedNodeId
    }));
  }, [graph.nodes, selectedNodeId]);

  const edges = useMemo<Edge[]>(() => {
    return graph.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      animated: !reducedMotion && Boolean(pulsedNodeId),
      className: "pulse-edge"
    }));
  }, [graph.edges, pulsedNodeId, reducedMotion]);

  useGSAP(() => {
    if (!pulsedNodeId || reducedMotion || viewMode !== "ops") return;
    const target = containerRef.current?.querySelector(`[data-layer-id="${pulsedNodeId}"]`);
    if (!target) return;
    gsap.fromTo(
      target,
      { scale: 1, boxShadow: "0 0 0 rgba(79, 209, 197, 0)" },
      {
        scale: 1.06,
        boxShadow: "0 0 30px rgba(79, 209, 197, 0.55)",
        yoyo: true,
        repeat: 1,
        duration: 0.28,
        ease: "power2.out"
      }
    );
  }, { dependencies: [pulsedNodeId, reducedMotion, viewMode], scope: containerRef });

  const handleSelectLayer = (nodeId: string) => {
    const node = graph.nodes.find((item) => item.id === nodeId);
    if (node) onSelect(node);
  };

  return (
    <section className="graph-panel" ref={containerRef}>
      <div className="panel-heading">
        <div>
          <h2>{viewMode === "ops" ? "Operator Graph" : "Neural Network"}</h2>
          <span>{viewMode === "ops" ? "Netron-like structure with training pulses" : "Layered neurons with activation intensity"}</span>
        </div>
        <div className="view-tabs" aria-label="graph view">
          <button className={viewMode === "ops" ? "active" : ""} onClick={() => setViewMode("ops")} type="button">Ops</button>
          <button className={viewMode === "neurons" ? "active" : ""} onClick={() => setViewMode("neurons")} type="button">Neurons</button>
        </div>
      </div>
      {viewMode === "ops" ? (
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          minZoom={0.35}
          onNodeClick={(_, node) => onSelect(node.data)}
        >
          <Background color="#263244" />
          <MiniMap pannable zoomable />
          <Controls />
        </ReactFlow>
      ) : (
        <NeuralNetworkView graph={graph} probabilities={probabilities} pulsedNodeId={pulsedNodeId} onSelectLayer={handleSelectLayer} />
      )}
    </section>
  );
}
