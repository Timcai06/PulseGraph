import { describe, expect, it } from "vitest";
import type { GraphNode, LayerSnapshot } from "../api/client";
import { deriveLayerHealth, formatNodeShape, formatParamCount } from "./layerHealth";

const node: GraphNode = {
  id: "net.3",
  label: "net.3",
  kind: "Linear",
  input_shape: [16, 128],
  output_shape: [16, 10],
  param_count: 1290,
  confidence: "trusted",
  metadata: {}
};

describe("layer health", () => {
  it("flags a possible dead layer from high activation sparsity", () => {
    const snapshot: LayerSnapshot = {
      layer_id: "net.3",
      activation_sparsity: 0.98,
      gradient_norm: 0.1
    };

    expect(deriveLayerHealth(node, snapshot)).toMatchObject({
      severity: "warning",
      label: "sparse activation",
      detail: "Activation sparsity is high; possible dead layer."
    });
  });

  it("flags a possible vanishing gradient from near-zero gradient norm", () => {
    const snapshot: LayerSnapshot = {
      layer_id: "net.3",
      activation_sparsity: 0.1,
      gradient_norm: 1e-8
    };

    expect(deriveLayerHealth(node, snapshot)).toMatchObject({
      severity: "warning",
      label: "low gradient",
      detail: "Gradient norm is near zero; possible vanishing gradient."
    });
  });

  it("formats node shape and parameter count for compact node metadata", () => {
    expect(formatNodeShape(node)).toBe("[16x128] -> [16x10]");
    expect(formatParamCount(node.param_count)).toBe("1.3k");
  });
});
