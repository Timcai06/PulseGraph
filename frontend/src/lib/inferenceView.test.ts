import { describe, expect, it } from "vitest";
import {
  chartProbabilityRows,
  classificationOutputFromPrediction,
  displayClassName,
  inferenceOutputKind,
  normalizeImageShape,
  structuredOutputRows,
  topProbabilityRows
} from "./inferenceView";

describe("inference view helpers", () => {
  it("uses class names when available and falls back to numeric labels", () => {
    expect(displayClassName(1, ["cat", "dog"])).toBe("dog");
    expect(displayClassName(3, ["cat", "dog"])).toBe("3");
    expect(displayClassName(2)).toBe("2");
  });

  it("returns top probability rows with readable labels", () => {
    expect(topProbabilityRows([0.1, 0.7, 0.2], ["low", "mid", "high"])).toEqual([
      { index: 1, label: "mid", value: 0.7 },
      { index: 2, label: "high", value: 0.2 },
      { index: 0, label: "low", value: 0.1 }
    ]);
  });

  it("keeps small probability charts in class order", () => {
    expect(chartProbabilityRows([0.2, 0.5, 0.3], ["red", "green", "blue"]).map((row) => row.label)).toEqual([
      "red",
      "green",
      "blue"
    ]);
  });

  it("limits large probability charts to the top ten classes", () => {
    const probabilities = Array.from({ length: 24 }, (_, index) => index / 100);
    const rows = chartProbabilityRows(probabilities);

    expect(rows).toHaveLength(10);
    expect(rows[0]).toEqual({ index: 23, label: "23", value: 0.23 });
    expect(rows[9]).toEqual({ index: 14, label: "14", value: 0.14 });
  });

  it("resolves classification output from the task contract before legacy fields", () => {
    const view = classificationOutputFromPrediction({
      task: "classification",
      output: {
        kind: "classification",
        label: 2,
        prediction: 1,
        confidence: 0.82,
        probabilities: [0.1, 0.82, 0.08],
        class_names: ["red", "green", "blue"]
      },
      sample_index: 0,
      label: 0,
      prediction: 0,
      weights: "trained",
      sample_source: "probe",
      class_names: ["old"],
      image_shape: [3, 2, 2],
      image_pixels: Array(12).fill(0),
      probabilities: [0.9, 0.05, 0.05],
      graph: { nodes: [], edges: [] },
      layers: []
    });

    expect(view).toEqual({
      label: 2,
      prediction: 1,
      confidence: 0.82,
      probabilities: [0.1, 0.82, 0.08],
      classNames: ["red", "green", "blue"]
    });
  });

  it("summarizes non-classification outputs for the fallback renderer", () => {
    const prediction = {
      task: "detection",
      output: { kind: "detection", boxes: [[0, 0, 4, 4]], score: 0.93123, model: "tiny-detector" },
      sample_index: 0,
      label: 0,
      prediction: 0,
      weights: "trained" as const,
      sample_source: "probe" as const,
      image_shape: [3, 2, 2],
      image_pixels: Array(12).fill(0),
      probabilities: [],
      graph: { nodes: [], edges: [] },
      layers: []
    };

    expect(inferenceOutputKind(prediction)).toBe("detection");
    expect(classificationOutputFromPrediction(prediction)).toBeUndefined();
    expect(structuredOutputRows(prediction.output)).toEqual([
      { key: "boxes", value: "1 items" },
      { key: "score", value: "0.9312" },
      { key: "model", value: "tiny-detector" }
    ]);
  });

  it("normalizes image shapes to channel-height-width", () => {
    expect(normalizeImageShape([3, 32, 32], 3 * 32 * 32)).toEqual([3, 32, 32]);
    expect(normalizeImageShape([28, 28], 28 * 28)).toEqual([1, 28, 28]);
    expect(normalizeImageShape(undefined, 784)).toEqual([1, 28, 28]);
    expect(normalizeImageShape([2, 2, 2], 8)).toBeUndefined();
  });
});
