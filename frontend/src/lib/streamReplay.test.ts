import { describe, expect, it } from "vitest";
import { RUN_REPLAY_EVENT_DELAY_MS, replayDelayMs } from "./streamReplay";

describe("replayDelayMs", () => {
  it("paces recorded run events so streaming replays visibly instead of all at once", () => {
    expect(replayDelayMs(0, "train-abc")).toBe(0);
    expect(replayDelayMs(3, "train-abc")).toBe(3 * RUN_REPLAY_EVENT_DELAY_MS);
  });

  it("does not delay demo events because the demo stream is already paced by the server", () => {
    expect(replayDelayMs(3)).toBe(0);
  });
});
