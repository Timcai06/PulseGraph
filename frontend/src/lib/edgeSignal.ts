import type { GraphEdge, ModelGraph } from "../api/client";

/**
 * Groups edges into sequential levels for the forward-pass signal sweep.
 * An edge's level is the longest-path depth of its source node, so parallel
 * branches light up together and the sweep reaches every node only after
 * all of its inputs have arrived (Kahn topological order).
 */
export function orderEdgesForSweep(graph: ModelGraph): GraphEdge[][] {
  const incoming = new Map<string, number>();
  const outgoing = new Map<string, GraphEdge[]>();
  for (const node of graph.nodes) incoming.set(node.id, 0);
  for (const edge of graph.edges) {
    incoming.set(edge.target, (incoming.get(edge.target) ?? 0) + 1);
    const list = outgoing.get(edge.source) ?? [];
    list.push(edge);
    outgoing.set(edge.source, list);
  }

  const depth = new Map<string, number>();
  const queue: string[] = [];
  for (const node of graph.nodes) {
    if ((incoming.get(node.id) ?? 0) === 0) {
      depth.set(node.id, 0);
      queue.push(node.id);
    }
  }

  const remaining = new Map(incoming);
  while (queue.length) {
    const id = queue.shift()!;
    for (const edge of outgoing.get(id) ?? []) {
      const nextDepth = Math.max(depth.get(edge.target) ?? 0, (depth.get(id) ?? 0) + 1);
      depth.set(edge.target, nextDepth);
      const left = (remaining.get(edge.target) ?? 0) - 1;
      remaining.set(edge.target, left);
      if (left === 0) queue.push(edge.target);
    }
  }

  const levels: GraphEdge[][] = [];
  for (const edge of graph.edges) {
    const level = depth.get(edge.source) ?? 0;
    (levels[level] ??= []).push(edge);
  }
  return levels.filter((level) => level !== undefined && level.length > 0);
}

/** Edges that feed the given node; used for the per-snapshot comet flash. */
export function incomingEdges(graph: ModelGraph, nodeId: string): GraphEdge[] {
  return graph.edges.filter((edge) => edge.target === nodeId);
}
