import { describe, expect, it } from "vitest";
import {
  chartProbabilityRows,
  classificationOutputFromPrediction,
  describeDetectionSummary,
  detectionOutputFromOutput,
  displayClassName,
  inferenceSampleCaption,
  inferenceOutputKind,
  normalizeImageShape,
  resolveDetectionOverlay,
  resolveInferenceRenderer,
  structuredOutputRows,
  topProbabilityRows
} from "./inferenceView";
import type { PredictionResponse } from "../api/types";

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

  it("keeps the rich classification view when legacy fields are absent", () => {
    const prediction: PredictionResponse = {
      task: "classification",
      output: {
        kind: "classification",
        label: 1,
        prediction: 2,
        confidence: 0.7,
        probabilities: [0.1, 0.2, 0.7],
        class_names: ["red", "green", "blue"]
      },
      sample_index: 0,
      weights: "trained",
      sample_source: "probe",
      image_shape: [3, 2, 2],
      image_pixels: Array(12).fill(0),
      graph: { nodes: [], edges: [] },
      layers: []
    };

    expect(classificationOutputFromPrediction(prediction)).toEqual({
      label: 1,
      prediction: 2,
      confidence: 0.7,
      probabilities: [0.1, 0.2, 0.7],
      classNames: ["red", "green", "blue"]
    });
  });

  it("parses detection outputs for the overlay renderer", () => {
    const detection = detectionOutputFromOutput({
      kind: "detection",
      boxes: [[1, 2, 6, 7]],
      labels: [1],
      scores: [0.93123],
      label_names: ["square"]
    });

    expect(detection).toEqual({
      boxes: [
        {
          index: 0,
          label: 1,
          labelName: "square",
          score: 0.93123,
          coordinates: [1, 2, 6, 7]
        }
      ],
      totalCount: 1,
      truncated: false
    });
    expect(describeDetectionSummary(detection!)).toBe("square · 93.1%");
  });

  it("resolves detection output through renderer semantics", () => {
    const prediction: PredictionResponse = {
      task: "detection",
      output: {
        kind: "detection",
        boxes: [[0, 0, 4, 4]],
        labels: [1],
        scores: [0.93123],
        label_names: ["square"]
      },
      sample_index: 0,
      weights: "trained",
      sample_source: "probe",
      image_shape: [3, 2, 2],
      image_pixels: Array(12).fill(0),
      graph: { nodes: [], edges: [] },
      layers: []
    };

    expect(inferenceOutputKind(prediction)).toBe("detection");
    expect(classificationOutputFromPrediction(prediction)).toBeUndefined();
    expect(resolveInferenceRenderer(prediction)).toEqual({
      renderer: "detection",
      kind: "detection",
      detection: {
        boxes: [
          {
            index: 0,
            label: 1,
            labelName: "square",
            score: 0.93123,
            coordinates: [0, 0, 4, 4]
          }
        ],
        totalCount: 1,
        truncated: false
      }
    });
  });

  it("uses renderer hints to recover detection overlays for preview samples", () => {
    const detection = resolveDetectionOverlay(
      {
        boxes: [[1, 1, 6, 6]],
        labels: [1],
        scores: [],
        label_names: ["square"]
      },
      { rendererHint: "box_overlay", task: "detection" }
    );

    expect(detection).toEqual({
      boxes: [
        {
          index: 0,
          label: 1,
          labelName: "square",
          score: undefined,
          coordinates: [1, 1, 6, 6]
        }
      ],
      totalCount: 1,
      truncated: false
    });
    expect(
      inferenceSampleCaption({
        output: { boxes: [[1, 1, 6, 6]], labels: [1], scores: [], label_names: ["square"] },
        rendererHint: "box_overlay",
        task: "detection"
      })
    ).toBe("square");
  });

  it("bounds detection DOM work while preserving the declared total", () => {
    const boxes = Array.from({ length: 140 }, (_, index) => [index, index, index + 1, index + 1]);
    const detection = detectionOutputFromOutput({
      kind: "detection",
      boxes,
      labels: Array(140).fill(1),
      total_detections: 140,
      truncated: true
    });

    expect(detection?.boxes).toHaveLength(100);
    expect(detection?.totalCount).toBe(140);
    expect(detection?.truncated).toBe(true);
    expect(describeDetectionSummary(detection!)).toBe("100 shown of 140 objects");
  });

  it("keeps unknown structured outputs in the fallback renderer", () => {
    const prediction: PredictionResponse = {
      task: "segmentation",
      output: { kind: "segmentation", mask_shape: [64, 64], codec: "rle", sparse: true },
      sample_index: 0,
      weights: "trained",
      sample_source: "probe",
      image_shape: [3, 8, 8],
      image_pixels: Array(192).fill(0),
      graph: { nodes: [], edges: [] },
      layers: []
    };

    expect(resolveInferenceRenderer(prediction)).toEqual({
      renderer: "structured",
      kind: "segmentation",
      rows: [
        { key: "mask_shape", value: "2 items" },
        { key: "codec", value: "rle" },
        { key: "sparse", value: "true" }
      ]
    });
    expect(structuredOutputRows(prediction.output)).toEqual([
      { key: "mask_shape", value: "2 items" },
      { key: "codec", value: "rle" },
      { key: "sparse", value: "true" }
    ]);
  });

  it("normalizes image shapes to channel-height-width", () => {
    expect(normalizeImageShape([3, 32, 32], 3 * 32 * 32)).toEqual([3, 32, 32]);
    expect(normalizeImageShape([28, 28], 28 * 28)).toEqual([1, 28, 28]);
    expect(normalizeImageShape(undefined, 784)).toEqual([1, 28, 28]);
    expect(normalizeImageShape([2, 2, 2], 8)).toBeUndefined();
  });
});
