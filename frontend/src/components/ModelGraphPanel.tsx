import { memo, useMemo, useRef, useState } from "react";
import { Handle, Position, ReactFlow, Background, Controls, type Edge, type Node, type NodeProps } from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import type { GraphNode, ModelGraph } from "../api/client";
import { useReducedMotion } from "../hooks/useReducedMotion";
import { displayGraph } from "../lib/graphView";
import { NeuralNetworkView } from "./NeuralNetworkView";

gsap.registerPlugin(useGSAP);

const NODE_WIDTH = 178;
const NODE_HEIGHT = 96;

function layoutPositions(graph: ModelGraph): Record<string, { x: number; y: number }> {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "LR", nodesep: 36, ranksep: 64 });
  g.setDefaultEdgeLabel(() => ({}));
  for (const node of graph.nodes) g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  for (const edge of graph.edges) g.setEdge(edge.source, edge.target);
  dagre.layout(g);
  const positions: Record<string, { x: number; y: number }> = {};
  for (const node of graph.nodes) {
    const placed = g.node(node.id);
    positions[node.id] = placed
      ? { x: placed.x - NODE_WIDTH / 2, y: placed.y - NODE_HEIGHT / 2 }
      : { x: 0, y: 0 };
  }
  return positions;
}

const PulseNode = memo(({ data, selected }: NodeProps<Node<GraphNode>>) => {
  return (
    <div className={`model-node ${selected ? "selected" : ""}`} data-layer-id={data.id}>
      <Handle type="target" position={Position.Left} />
      <div className="node-kind">{data.kind}</div>
      <div className="node-label">{data.id}</div>
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
  const reducedMotion = useReducedMotion();
  const [viewMode, setViewMode] = useState<"ops" | "neurons">("ops");
  const visibleGraph = useMemo(() => displayGraph(graph), [graph]);

  const positions = useMemo(() => layoutPositions(visibleGraph), [visibleGraph]);
  const nodes = useMemo<Node<GraphNode>[]>(() => {
    return visibleGraph.nodes.map((node) => ({
      id: node.id,
      type: "pulse",
      position: positions[node.id] ?? { x: 0, y: 0 },
      data: node,
      selected: node.id === selectedNodeId
    }));
  }, [visibleGraph.nodes, positions, selectedNodeId]);

  const edges = useMemo<Edge[]>(() => {
    return visibleGraph.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      animated: !reducedMotion && Boolean(pulsedNodeId),
      className: "pulse-edge"
    }));
  }, [visibleGraph.edges, pulsedNodeId, reducedMotion]);

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
    const node = visibleGraph.nodes.find((item) => item.id === nodeId);
    if (node) onSelect(node);
  };

  return (
    <section className="graph-panel" ref={containerRef}>
      <div className="panel-heading">
        <div>
          <h2>{viewMode === "ops" ? "Operator Graph" : "Neural Network"}</h2>
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
          fitViewOptions={{ padding: 0.24 }}
          minZoom={0.35}
          nodesDraggable={false}
          onNodeClick={(_, node) => onSelect(node.data)}
        >
          <Background color="#263244" />
          <Controls />
        </ReactFlow>
      ) : (
        <NeuralNetworkView graph={visibleGraph} probabilities={probabilities} pulsedNodeId={pulsedNodeId} onSelectLayer={handleSelectLayer} />
      )}
    </section>
  );
}
