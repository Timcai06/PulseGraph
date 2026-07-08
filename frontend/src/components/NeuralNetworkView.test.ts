import { describe, expect, it } from "vitest";
import type { ModelGraph } from "../api/client";
import { buildNeuralLayers } from "./NeuralNetworkView";

const demoGraph: ModelGraph = {
  nodes: [
    { id: "input", label: "Input", kind: "Input", output_shape: [1, 28, 28], param_count: 0, confidence: "trusted", metadata: {} },
    { id: "flatten", label: "Flatten", kind: "Flatten", output_shape: [784], param_count: 0, confidence: "trusted", metadata: {} },
    { id: "linear1", label: "Linear 784 -> 128", kind: "Linear", output_shape: [128], param_count: 100480, confidence: "trusted", metadata: {} },
    { id: "relu1", label: "ReLU", kind: "ReLU", output_shape: [128], param_count: 0, confidence: "trusted", metadata: {} },
    { id: "linear2", label: "Linear 128 -> 10", kind: "Linear", output_shape: [10], param_count: 1290, confidence: "trusted", metadata: {} },
    { id: "softmax", label: "Softmax", kind: "Softmax", output_shape: [10], param_count: 0, confidence: "trusted", metadata: {} }
  ],
  edges: []
};

describe("buildNeuralLayers", () => {
  it("maps a model graph into model-layer neural layers without showing the input placeholder", () => {
    const layers = buildNeuralLayers({
      graph: demoGraph,
      probabilities: [0.05, 0.7, 0.25],
      pulsedNodeId: "linear1"
    });

    expect(layers.map((layer) => layer.role)).toEqual(["hidden", "output"]);
    expect(layers[0].sourceNodeId).toBe("linear1");
    expect(layers[0].active).toBe(true);
    expect(layers[1].neurons).toHaveLength(3);
    expect(layers[1].neurons[1].intensity).toBeCloseTo(0.7);
    expect(layers[1].neurons[1].strongest).toBe(true);
    expect(layers[0].neurons[0].activationOrder).toBeLessThan(layers[1].neurons[0].activationOrder);
  });
});
