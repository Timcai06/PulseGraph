import { describe, expect, it } from "vitest";
import { runContractFromConfig } from "./runContract";

describe("runContractFromConfig", () => {
  it("hydrates detection renderer and metrics from persisted run config", () => {
    expect(
      runContractFromConfig({
        task: "Detection",
        output_schema: { kind: "detection", renderer: "box_overlay" },
        metric_schema: { primary: "mean_iou", monitors: ["loss", "mean_iou"] }
      })
    ).toEqual({
      task: "detection",
      outputSchema: { kind: "detection", renderer: "box_overlay" },
      metricSchema: { primary: "mean_iou", monitors: ["loss", "mean_iou"] }
    });
  });

  it("ignores malformed schema values without losing a valid task", () => {
    expect(runContractFromConfig({ task: "classification", output_schema: "invalid", metric_schema: [] })).toEqual({
      task: "classification",
      outputSchema: undefined,
      metricSchema: undefined
    });
  });
});
