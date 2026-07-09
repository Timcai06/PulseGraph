import { describe, expect, it } from "vitest";
import type { ModelGraph } from "../api/client";
import { incomingEdges, orderEdgesForSweep } from "./edgeSignal";

function node(id: string) {
  return { id, label: id, kind: "Linear", param_count: 0, confidence: "safe" as const, metadata: {} };
}

function edge(source: string, target: string) {
  return { id: `${source}->${target}`, source, target };
}

describe("orderEdgesForSweep", () => {
  it("orders a linear chain one edge per level", () => {
    const graph: ModelGraph = {
      nodes: [node("a"), node("b"), node("c")],
      edges: [edge("b", "c"), edge("a", "b")]
    };
    const levels = orderEdgesForSweep(graph);
    expect(levels.map((level) => level.map((item) => item.id))).toEqual([["a->b"], ["b->c"]]);
  });

  it("groups parallel branches into the same level and joins after both arrive", () => {
    const graph: ModelGraph = {
      nodes: [node("in"), node("left"), node("right"), node("out")],
      edges: [edge("in", "left"), edge("in", "right"), edge("left", "out"), edge("right", "out")]
    };
    const levels = orderEdgesForSweep(graph);
    expect(levels).toHaveLength(2);
    expect(levels[0].map((item) => item.id).sort()).toEqual(["in->left", "in->right"]);
    expect(levels[1].map((item) => item.id).sort()).toEqual(["left->out", "right->out"]);
  });

  it("fires a skip edge earlier than the deep path, mirroring real arrival order", () => {
    // in -> skip -> out  plus  in -> mid -> deep -> out: the skip edge is level 1,
    // the deep->out edge is level 2
    const graph: ModelGraph = {
      nodes: [node("in"), node("skip"), node("mid"), node("deep"), node("out")],
      edges: [edge("in", "skip"), edge("in", "mid"), edge("mid", "deep"), edge("skip", "out"), edge("deep", "out")]
    };
    const levels = orderEdgesForSweep(graph);
    expect(levels[1].map((item) => item.id)).toContain("skip->out");
    expect(levels[2].map((item) => item.id)).toContain("deep->out");
  });

  it("returns no levels for an empty graph", () => {
    expect(orderEdgesForSweep({ nodes: [], edges: [] })).toEqual([]);
  });
});

describe("incomingEdges", () => {
  it("finds only edges targeting the node", () => {
    const graph: ModelGraph = {
      nodes: [node("a"), node("b"), node("c")],
      edges: [edge("a", "b"), edge("b", "c")]
    };
    expect(incomingEdges(graph, "c").map((item) => item.id)).toEqual(["b->c"]);
    expect(incomingEdges(graph, "a")).toEqual([]);
  });
});
