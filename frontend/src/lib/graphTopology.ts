import type { ModelGraph } from "../api/client";

export type NodeTopology = {
  depth: number;
  lane: number;
  group: string;
  incoming: number;
  outgoing: number;
  role: "linear" | "branch" | "merge" | "branch-merge";
};

export type EdgeTopology = {
  kind: "sequential" | "skip" | "merge";
};

export type GraphTopology = {
  nodes: Record<string, NodeTopology>;
  edges: Record<string, EdgeTopology>;
  hasBranching: boolean;
  hasSkipConnections: boolean;
  maxDepth: number;
};

function groupName(id: string, label: string, metadata: Record<string, unknown>): string {
  const modulePath = metadata.module_path ?? metadata.module ?? metadata.scope;
  if (typeof modulePath === "string" && modulePath.trim()) return modulePath.split(".")[0];
  const source = label || id;
  return source.includes(".") ? source.split(".")[0] : source.split(/[_:-]/)[0] || source;
}

export function deriveGraphTopology(graph: ModelGraph): GraphTopology {
  const incoming = new Map<string, string[]>();
  const outgoing = new Map<string, string[]>();
  for (const node of graph.nodes) {
    incoming.set(node.id, []);
    outgoing.set(node.id, []);
  }
  for (const edge of graph.edges) {
    incoming.get(edge.target)?.push(edge.source);
    outgoing.get(edge.source)?.push(edge.target);
  }

  const depths = new Map<string, number>();
  const visiting = new Set<string>();
  const depthFor = (nodeId: string): number => {
    if (depths.has(nodeId)) return depths.get(nodeId) ?? 0;
    if (visiting.has(nodeId)) return 0;
    visiting.add(nodeId);
    const parents = incoming.get(nodeId) ?? [];
    const depth = parents.length ? Math.max(...parents.map((parent) => depthFor(parent) + 1)) : 0;
    visiting.delete(nodeId);
    depths.set(nodeId, depth);
    return depth;
  };
  for (const node of graph.nodes) depthFor(node.id);

  const laneCounters = new Map<number, number>();
  const nodes: Record<string, NodeTopology> = {};
  for (const node of graph.nodes) {
    const depth = depths.get(node.id) ?? 0;
    const lane = laneCounters.get(depth) ?? 0;
    laneCounters.set(depth, lane + 1);
    const inCount = incoming.get(node.id)?.length ?? 0;
    const outCount = outgoing.get(node.id)?.length ?? 0;
    const role = outCount > 1 && inCount > 1 ? "branch-merge" : outCount > 1 ? "branch" : inCount > 1 ? "merge" : "linear";
    nodes[node.id] = {
      depth,
      lane,
      group: groupName(node.id, node.label, node.metadata),
      incoming: inCount,
      outgoing: outCount,
      role
    };
  }

  const edges: Record<string, EdgeTopology> = {};
  for (const edge of graph.edges) {
    const sourceDepth = depths.get(edge.source) ?? 0;
    const targetDepth = depths.get(edge.target) ?? 0;
    edges[edge.id] = {
      kind: targetDepth - sourceDepth > 1 ? "skip" : (incoming.get(edge.target)?.length ?? 0) > 1 ? "merge" : "sequential"
    };
  }

  return {
    nodes,
    edges,
    hasBranching: Object.values(nodes).some((node) => node.role === "branch" || node.role === "branch-merge"),
    hasSkipConnections: Object.values(edges).some((edge) => edge.kind === "skip"),
    maxDepth: Math.max(0, ...Object.values(nodes).map((node) => node.depth))
  };
}
