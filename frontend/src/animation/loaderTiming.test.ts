import { describe, expect, it } from "vitest";
import {
  displayedProgressValue,
  introCharGroup,
  introRiseStagger,
  progressDampFactor,
  stepDisplayedProgress
} from "./loaderTiming";

describe("introCharGroup", () => {
  it("splits 'PulseGraph.' into P | ulse | Graph | .", () => {
    const groups = "PulseGraph.".split("").map((_, i) => introCharGroup(i));
    expect(groups).toEqual([0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3]);
  });
});

describe("introRiseStagger", () => {
  it("beats groups 0.16s apart with 0.018s per-char refinement", () => {
    expect(introRiseStagger(0, 0)).toBe(0);
    expect(introRiseStagger(1, 1)).toBeCloseTo(0.178);
    expect(introRiseStagger(3, 10)).toBeCloseTo(0.66);
  });

  it("orders every char of the title strictly by index", () => {
    const delays = "PulseGraph.".split("").map((_, i) => introRiseStagger(introCharGroup(i), i));
    const sorted = [...delays].sort((a, b) => a - b);
    expect(delays).toEqual(sorted);
    expect(new Set(delays).size).toBe(delays.length);
  });
});

describe("progress damping", () => {
  it("tracks slowly before ready and snaps faster after", () => {
    expect(progressDampFactor(false)).toBeLessThan(progressDampFactor(true));
    expect(stepDisplayedProgress(0, 1, false)).toBeCloseTo(0.075);
    expect(stepDisplayedProgress(0, 1, true)).toBeCloseTo(0.18);
  });

  it("converges monotonically toward the target", () => {
    let displayed = 0;
    for (let i = 0; i < 60; i++) {
      const next = stepDisplayedProgress(displayed, 1, true);
      expect(next).toBeGreaterThan(displayed);
      displayed = next;
    }
    expect(displayed).toBeGreaterThan(0.99);
  });
});

describe("displayedProgressValue", () => {
  it("never shows 100 before ready", () => {
    expect(displayedProgressValue(0.999, false)).toBe(99);
    expect(displayedProgressValue(1, false)).toBe(99);
  });

  it("rounds up to close the gap once ready, capped at 100", () => {
    expect(displayedProgressValue(0.991, true)).toBe(100);
    expect(displayedProgressValue(0.5, true)).toBe(50);
    expect(displayedProgressValue(1.2, true)).toBe(100);
  });
});
