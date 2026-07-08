import { describe, expect, it } from "vitest";
import type { ModelGraph } from "../api/client";
import { displayGraph, firstDisplayNode } from "./graphView";

describe("displayGraph", () => {
  it("removes fx input placeholders from the visible model graph", () => {
    const graph: ModelGraph = {
      nodes: [
        { id: "input", label: "input", kind: "Input", param_count: 0, confidence: "trusted", metadata: {} },
        { id: "net.0", label: "Flatten", kind: "Flatten", param_count: 0, confidence: "trusted", metadata: {} },
        { id: "net.1", label: "Linear", kind: "Linear", param_count: 10, confidence: "trusted", metadata: {} }
      ],
      edges: [
        { id: "input->net.0", source: "input", target: "net.0" },
        { id: "net.0->net.1", source: "net.0", target: "net.1" }
      ]
    };

    const visible = displayGraph(graph);

    expect(visible.nodes.map((node) => node.id)).toEqual(["net.0", "net.1"]);
    expect(visible.edges.map((edge) => edge.id)).toEqual(["net.0->net.1"]);
  });

  it("selects the first real layer instead of the fx input placeholder", () => {
    const graph: ModelGraph = {
      nodes: [
        { id: "input", label: "input", kind: "Input", param_count: 0, confidence: "trusted", metadata: {} },
        { id: "flatten", label: "Flatten", kind: "Flatten", param_count: 0, confidence: "trusted", metadata: {} }
      ],
      edges: [{ id: "input->flatten", source: "input", target: "flatten" }]
    };

    expect(firstDisplayNode(graph)?.id).toBe("flatten");
  });

  it("prefers the first parameterized layer for the default inspector selection", () => {
    const graph: ModelGraph = {
      nodes: [
        { id: "input", label: "input", kind: "Input", param_count: 0, confidence: "trusted", metadata: {} },
        { id: "flatten", label: "Flatten", kind: "Flatten", param_count: 0, confidence: "trusted", metadata: {} },
        { id: "linear", label: "Linear", kind: "Linear", param_count: 100480, confidence: "trusted", metadata: {} }
      ],
      edges: [
        { id: "input->flatten", source: "input", target: "flatten" },
        { id: "flatten->linear", source: "flatten", target: "linear" }
      ]
    };

    expect(firstDisplayNode(graph)?.id).toBe("linear");
  });
});
