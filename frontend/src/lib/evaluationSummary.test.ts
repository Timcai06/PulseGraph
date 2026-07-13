import { describe, expect, it } from "vitest";
import type { RunDetail } from "../api/client";
import { evaluationSnapshot, topFailureRoutes } from "./evaluationSummary";

const detail = {
  config: { class_names: ["cat", "dog", "bird"] },
  evidence: [{
    kind: "evaluation_failures",
    split: "validation",
    evaluated_samples: 8,
    failures: [
      { sample_id: "a", label: 0, prediction: 1, confidence: 0.8 },
      { sample_id: "b", label: 0, prediction: 1, confidence: 0.7 },
      { sample_id: "c", label: 2, prediction: 1, confidence: 0.6 }
    ]
  }]
} as unknown as RunDetail;

describe("evaluationSummary", () => {
  it("normalizes recorded evaluation evidence", () => {
    expect(evaluationSnapshot(detail)).toMatchObject({ evaluated: 8, split: "validation" });
    expect(evaluationSnapshot(detail).failures).toHaveLength(3);
  });

  it("ranks repeated class confusion routes", () => {
    expect(topFailureRoutes(detail)).toEqual([
      { label: "cat", prediction: "dog", count: 2 },
      { label: "bird", prediction: "dog", count: 1 }
    ]);
  });
});
