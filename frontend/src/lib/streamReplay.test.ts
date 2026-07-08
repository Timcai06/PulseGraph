import { describe, expect, it } from "vitest";
import { replayDelayMs } from "./streamReplay";

describe("replayDelayMs", () => {
  it("does not add client-side delay because the backend owns stream pacing", () => {
    expect(replayDelayMs(0, "train-abc")).toBe(0);
    expect(replayDelayMs(3, "train-abc")).toBe(0);
    expect(replayDelayMs(3)).toBe(0);
  });
});
