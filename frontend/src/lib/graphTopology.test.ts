import { describe, expect, it } from "vitest";
import type { ModelGraph } from "../api/client";
import { deriveGraphTopology } from "./graphTopology";

describe("graph topology", () => {
  it("marks branch, merge, depth, and skip-connection semantics", () => {
    const graph: ModelGraph = {
      nodes: [
        { id: "stem", label: "features.stem", kind: "Conv2d", param_count: 1, confidence: "trusted", metadata: {} },
        { id: "left", label: "features.left", kind: "Conv2d", param_count: 1, confidence: "trusted", metadata: {} },
        { id: "right", label: "features.right", kind: "Conv2d", param_count: 1, confidence: "trusted", metadata: {} },
        { id: "merge", label: "head.merge", kind: "Add", param_count: 0, confidence: "trusted", metadata: {} },
        { id: "out", label: "head.out", kind: "Linear", param_count: 1, confidence: "trusted", metadata: {} }
      ],
      edges: [
        { id: "e1", source: "stem", target: "left" },
        { id: "e2", source: "stem", target: "right" },
        { id: "e3", source: "left", target: "merge" },
        { id: "e4", source: "right", target: "merge" },
        { id: "e5", source: "stem", target: "out" },
        { id: "e6", source: "merge", target: "out" }
      ]
    };

    const topology = deriveGraphTopology(graph);

    expect(topology.nodes.stem.role).toBe("branch");
    expect(topology.nodes.merge.role).toBe("merge");
    expect(topology.nodes.out.depth).toBeGreaterThan(topology.nodes.left.depth);
    expect(topology.edges.e5.kind).toBe("skip");
    expect(topology.hasBranching).toBe(true);
    expect(topology.hasSkipConnections).toBe(true);
  });
});
