import { describe, expect, it } from "vitest";
import type { ModelGraph } from "../api/client";
import {
  assessGhostEdge,
  deriveGraphPorts,
  ghostEdgeId,
  wouldCreateCycle,
  type GraphPort
} from "./graphPorts";

function graph(): ModelGraph {
  return {
    nodes: [
      {
        id: "conv",
        label: "features.conv",
        kind: "Conv2d",
        input_shape: [16, 1, 28, 28],
        output_shape: [16, 8, 24, 24],
        param_count: 80,
        confidence: "trusted",
        metadata: {}
      },
      {
        id: "relu",
        label: "features.relu",
        kind: "ReLU",
        input_shape: [16, 8, 24, 24],
        output_shape: [16, 8, 24, 24],
        param_count: 0,
        confidence: "trusted",
        metadata: {}
      },
      {
        id: "head",
        label: "classifier",
        kind: "Linear",
        input_shape: [16, 4608],
        output_shape: [16, 10],
        param_count: 46090,
        confidence: "trusted",
        metadata: {}
      }
    ],
    edges: [
      { id: "conv->relu", source: "conv", target: "relu" },
      { id: "relu->head", source: "relu", target: "head" }
    ]
  };
}

function port(nodeId: string, direction: GraphPort["direction"], shape?: number[] | null): GraphPort {
  return {
    id: `${nodeId}:${direction}`,
    nodeId,
    direction,
    shape,
    tensorName: `${nodeId}.${direction}`
  };
}

describe("graph ports", () => {
  it("derives input and output tensor ports from graph nodes", () => {
    const ports = deriveGraphPorts(graph());

    expect(ports.map((item) => item.id)).toContain("conv:input");
    expect(ports.map((item) => item.id)).toContain("conv:output");
    expect(ports.find((item) => item.id === "conv:input")?.shape).toEqual([16, 1, 28, 28]);
    expect(ports.find((item) => item.id === "conv:output")?.shape).toEqual([16, 8, 24, 24]);
  });

  it("assesses exact, caution, unknown, and invalid ghost edge compatibility", () => {
    expect(assessGhostEdge(graph(), port("conv", "output", [16, 8]), port("relu", "input", [16, 8])).status).toBe("compatible");
    expect(assessGhostEdge(graph(), port("conv", "output", [32, 8]), port("relu", "input", [16, 8])).status).toBe("caution");
    expect(assessGhostEdge(graph(), port("conv", "output", undefined), port("relu", "input", [16, 8])).status).toBe("unknown");
    expect(assessGhostEdge(graph(), port("conv", "input", [16, 8]), port("relu", "input", [16, 8])).status).toBe("incompatible");
    expect(assessGhostEdge(graph(), port("conv", "output", [16, 8]), port("head", "input", [16, 10])).status).toBe("incompatible");
  });

  it("detects feed-forward cycles before creating ghost edges", () => {
    expect(wouldCreateCycle(graph(), "head", "conv")).toBe(true);
    expect(wouldCreateCycle(graph(), "conv", "head")).toBe(false);
  });

  it("creates stable ghost edge ids without touching real graph edges", () => {
    const id = ghostEdgeId("conv:output", "head:input");

    expect(id).toBe("ghost:conv:output->head:input");
    expect(graph().edges.map((edge) => edge.id)).not.toContain(id);
  });
});
