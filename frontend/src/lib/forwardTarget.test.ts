import { describe, expect, it } from "vitest";
import { describeForwardTarget, resolveForwardTarget } from "./forwardTarget";

describe("resolveForwardTarget", () => {
  it("uses the current run target before falling back to demo forward", () => {
    expect(resolveForwardTarget({ runId: "source-abc", checkpointStep: 1 })).toEqual({
      mode: "run",
      runId: "source-abc",
      checkpointStep: 1
    });
    expect(resolveForwardTarget(undefined)).toEqual({ mode: "demo" });
  });

  it("labels the button target so users know whether it will run a demo or their upload", () => {
    expect(describeForwardTarget({ runId: "source-abc", checkpointStep: 1 })).toBe("Current run: source-abc");
    expect(describeForwardTarget(undefined)).toBe("No current run; demo forward will run.");
  });
});
