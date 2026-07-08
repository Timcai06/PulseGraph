import type { GraphNode, ModelGraph } from "../api/client";

export function isInputPlaceholder(node?: GraphNode): boolean {
  return Boolean(node && (node.kind === "Input" || node.id === "input"));
}

export function displayGraph(graph: ModelGraph): ModelGraph {
  const hidden = new Set(graph.nodes.filter(isInputPlaceholder).map((node) => node.id));
  if (!hidden.size) return graph;

  return {
    nodes: graph.nodes.filter((node) => !hidden.has(node.id)),
    edges: graph.edges
      .filter((edge) => !hidden.has(edge.source) && !hidden.has(edge.target))
      .map((edge) => ({ ...edge }))
  };
}

export function firstDisplayNode(graph: ModelGraph): GraphNode | undefined {
  const visible = displayGraph(graph).nodes;
  return visible.find((node) => node.param_count > 0) ?? visible[0];
}
