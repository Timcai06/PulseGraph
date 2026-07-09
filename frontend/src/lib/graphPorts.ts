import type { ModelGraph } from "../api/client";

export type GraphPort = {
  id: string;
  nodeId: string;
  direction: "input" | "output";
  shape?: number[] | null;
  tensorName?: string;
};

export type GhostEdgeStatus = "compatible" | "caution" | "incompatible" | "unknown";

export type GhostEdge = {
  id: string;
  sourcePortId: string;
  targetPortId: string;
  sourcePort: GraphPort;
  targetPort: GraphPort;
  status: GhostEdgeStatus;
  reasons: string[];
};

export function portId(nodeId: string, direction: GraphPort["direction"]) {
  return `${nodeId}:${direction}`;
}

export function ghostEdgeId(sourcePortId: string, targetPortId: string) {
  return `ghost:${sourcePortId}->${targetPortId}`;
}

export function deriveGraphPorts(graph: ModelGraph): GraphPort[] {
  return graph.nodes.flatMap((node) => [
    {
      id: portId(node.id, "input"),
      nodeId: node.id,
      direction: "input" as const,
      shape: node.input_shape,
      tensorName: `${node.label || node.id}.input`
    },
    {
      id: portId(node.id, "output"),
      nodeId: node.id,
      direction: "output" as const,
      shape: node.output_shape,
      tensorName: `${node.label || node.id}.output`
    }
  ]);
}

export function wouldCreateCycle(graph: ModelGraph, sourceNodeId: string, targetNodeId: string): boolean {
  if (sourceNodeId === targetNodeId) return true;
  const outgoing = new Map<string, string[]>();
  for (const node of graph.nodes) outgoing.set(node.id, []);
  for (const edge of graph.edges) outgoing.get(edge.source)?.push(edge.target);

  const seen = new Set<string>();
  const visit = (nodeId: string): boolean => {
    if (nodeId === sourceNodeId) return true;
    if (seen.has(nodeId)) return false;
    seen.add(nodeId);
    return (outgoing.get(nodeId) ?? []).some(visit);
  };
  return visit(targetNodeId);
}

function shapesEqual(left?: number[] | null, right?: number[] | null) {
  return Boolean(left?.length && right?.length && left.length === right.length && left.every((value, index) => value === right[index]));
}

function shapeAssessment(sourceShape?: number[] | null, targetShape?: number[] | null): { status: GhostEdgeStatus; reason: string } {
  if (!sourceShape?.length || !targetShape?.length) {
    return { status: "unknown", reason: "One side has unknown tensor shape." };
  }
  if (shapesEqual(sourceShape, targetShape)) {
    return { status: "compatible", reason: `Shapes match: [${sourceShape.join("x")}].` };
  }
  if (sourceShape.length === targetShape.length && sourceShape.slice(1).every((value, index) => value === targetShape[index + 1])) {
    return {
      status: "caution",
      reason: `Feature dimensions match, but batch differs: [${sourceShape.join("x")}] -> [${targetShape.join("x")}].`
    };
  }
  return {
    status: "incompatible",
    reason: `Shape mismatch: [${sourceShape.join("x")}] -> [${targetShape.join("x")}].`
  };
}

export function assessGhostEdge(graph: ModelGraph, sourcePort: GraphPort, targetPort: GraphPort): GhostEdge {
  const reasons: string[] = [];
  let status: GhostEdgeStatus = "compatible";

  if (sourcePort.direction !== "output" || targetPort.direction !== "input") {
    status = "incompatible";
    reasons.push("Composer only allows output ports to connect into input ports.");
  }

  if (wouldCreateCycle(graph, sourcePort.nodeId, targetPort.nodeId)) {
    status = "incompatible";
    reasons.push("This connection would create a feed-forward cycle.");
  }

  if (status !== "incompatible") {
    const shape = shapeAssessment(sourcePort.shape, targetPort.shape);
    status = shape.status;
    reasons.push(shape.reason);
  }

  return {
    id: ghostEdgeId(sourcePort.id, targetPort.id),
    sourcePortId: sourcePort.id,
    targetPortId: targetPort.id,
    sourcePort,
    targetPort,
    status,
    reasons
  };
}
