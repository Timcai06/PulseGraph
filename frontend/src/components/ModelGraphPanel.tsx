import { memo, useMemo, useRef, useState } from "react";
import { Handle, Position, ReactFlow, Controls, type Edge, type Node, type NodeProps } from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import { DrawSVGPlugin } from "gsap/DrawSVGPlugin";
import { MotionPathPlugin } from "gsap/MotionPathPlugin";
import type { GraphNode, ModelGraph } from "../api/client";
import { useReducedMotion } from "../hooks/useReducedMotion";
import { incomingEdges, orderEdgesForSweep } from "../lib/edgeSignal";
import { displayGraph } from "../lib/graphView";
import { NeuralNetworkView } from "./NeuralNetworkView";

gsap.registerPlugin(useGSAP, DrawSVGPlugin, MotionPathPlugin);

const SVG_NS = "http://www.w3.org/2000/svg";
const SIGNAL_EDGE_SECONDS = 0.42;

function edgePathData(container: HTMLElement, edgeId: string): string | null {
  const group = container.querySelector(`.react-flow__edge[data-id="${CSS.escape(edgeId)}"]`);
  const path = group?.querySelector<SVGPathElement>(".react-flow__edge-path");
  return path?.getAttribute("d") ?? null;
}

/**
 * Signals live in an overlay svg we own inside the React Flow viewport:
 * appending them into the React-managed edge groups doesn't survive re-renders
 * (React Flow rebuilds edge elements when the graph prop changes), while the
 * viewport element itself is stable and carries the pan/zoom transform.
 */
function ensureSignalLayer(container: HTMLElement): SVGSVGElement | null {
  const viewport = container.querySelector<HTMLElement>(".react-flow__viewport");
  if (!viewport) return null;
  let layer = viewport.querySelector<SVGSVGElement>(":scope > .signal-layer");
  if (!layer) {
    layer = document.createElementNS(SVG_NS, "svg");
    layer.setAttribute("class", "signal-layer");
    layer.setAttribute("width", "1");
    layer.setAttribute("height", "1");
    viewport.appendChild(layer);
  }
  return layer;
}

function pulseNode(container: HTMLElement, nodeId: string, strong = false) {
  const target = container.querySelector(`[data-layer-id="${CSS.escape(nodeId)}"]`);
  if (!target) return;
  gsap.fromTo(
    target,
    { scale: 1, boxShadow: "0 0 0 rgba(79, 209, 197, 0)" },
    {
      scale: strong ? 1.1 : 1.06,
      boxShadow: `0 0 ${strong ? 44 : 30}px rgba(79, 209, 197, ${strong ? 0.75 : 0.55})`,
      yoyo: true,
      repeat: 1,
      duration: strong ? 0.34 : 0.28,
      ease: "power2.out"
    }
  );
}

/** Comet along an edge: a glow trail drawn tip-first, optionally with an orb riding the path. */
function launchEdgeSignal(timeline: gsap.core.Timeline, layer: SVGSVGElement, pathData: string, at: number, withOrb: boolean) {
  const seconds = SIGNAL_EDGE_SECONDS;

  const glow = document.createElementNS(SVG_NS, "path");
  glow.setAttribute("d", pathData);
  glow.setAttribute("class", "signal-path");
  layer.appendChild(glow);
  timeline
    .fromTo(glow, { drawSVG: "0% 0%" }, { drawSVG: "0% 100%", duration: seconds, ease: "power1.in" }, at)
    .to(glow, { drawSVG: "100% 100%", duration: seconds * 0.55, ease: "power1.out", onComplete: () => glow.remove() }, at + seconds);

  if (!withOrb) return;
  const orb = document.createElementNS(SVG_NS, "circle");
  orb.setAttribute("class", "signal-orb");
  orb.setAttribute("r", "4.5");
  layer.appendChild(orb);
  timeline
    .fromTo(orb, { opacity: 0 }, { opacity: 1, duration: 0.08 }, at)
    .to(orb, { motionPath: { path: pathData }, duration: seconds, ease: "power1.in" }, at)
    .to(orb, { opacity: 0, duration: 0.14, onComplete: () => orb.remove() }, at + seconds - 0.04);
}

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
  /* increments on every applied forward pass; triggers the full-path signal sweep */
  forwardTick: number;
  onSelect: (node: GraphNode) => void;
};

export function ModelGraphPanel({ graph, selectedNodeId, pulsedNodeId, probabilities, forwardTick, onSelect }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sweepRef = useRef<gsap.core.Timeline | null>(null);
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
      className: "pulse-edge"
    }));
  }, [visibleGraph.edges]);

  // per-snapshot pulse during training: node glow + comet on its incoming edges
  useGSAP(() => {
    const container = containerRef.current;
    if (!pulsedNodeId || !container || reducedMotion || viewMode !== "ops") return;
    if (sweepRef.current?.isActive()) return; // the forward sweep already animates arrivals
    pulseNode(container, pulsedNodeId);
    const layer = ensureSignalLayer(container);
    if (!layer) return;
    const timeline = gsap.timeline();
    for (const edge of incomingEdges(visibleGraph, pulsedNodeId)) {
      const pathData = edgePathData(container, edge.id);
      if (pathData) launchEdgeSignal(timeline, layer, pathData, 0, false);
    }
  }, { dependencies: [pulsedNodeId, reducedMotion, viewMode], scope: containerRef });

  // forward pass: signal travels the whole graph, layer by layer, into the output
  useGSAP(() => {
    const container = containerRef.current;
    if (!forwardTick || !container || reducedMotion || viewMode !== "ops") return;
    sweepRef.current?.kill();
    container.querySelectorAll(".signal-path, .signal-orb").forEach((el) => el.remove());
    const layer = ensureSignalLayer(container);
    if (!layer) return;

    const levels = orderEdgesForSweep(visibleGraph);
    if (!levels.length) return;
    const timeline = gsap.timeline({ onComplete: () => (sweepRef.current = null) });
    let at = 0;
    for (const [index, level] of levels.entries()) {
      const arrived = new Set(level.map((edge) => edge.target));
      let launched = false;
      for (const edge of level) {
        const pathData = edgePathData(container, edge.id);
        if (!pathData) continue;
        launchEdgeSignal(timeline, layer, pathData, at, true);
        launched = true;
      }
      if (!launched) continue;
      const lastLevel = index === levels.length - 1;
      timeline.call(
        () => arrived.forEach((nodeId) => pulseNode(container, nodeId, lastLevel)),
        undefined,
        at + SIGNAL_EDGE_SECONDS
      );
      at += SIGNAL_EDGE_SECONDS * 0.92;
    }
    sweepRef.current = timeline;
  }, { dependencies: [forwardTick], scope: containerRef });

  const handleSelectLayer = (nodeId: string) => {
    const node = visibleGraph.nodes.find((item) => item.id === nodeId);
    if (node) onSelect(node);
  };

  return (
    <section className="graph-stage" ref={containerRef}>
      <div className="stage-toolbar">
        <span className="stage-title">{viewMode === "ops" ? "Operator Graph" : "Neural Network"}</span>
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
          <Controls />
        </ReactFlow>
      ) : (
        <NeuralNetworkView graph={visibleGraph} probabilities={probabilities} pulsedNodeId={pulsedNodeId} onSelectLayer={handleSelectLayer} />
      )}
    </section>
  );
}
