import { describe, expect, it } from "vitest";
import type { RunSummary } from "../api/client";
import { splitRunBuckets } from "./runViews";

function run(run_id: string, completed: boolean): RunSummary {
  return {
    run_id,
    completed,
    created_at: 1,
    last_event_at: 2,
    event_count: 3,
    last_step: 4
  };
}

describe("splitRunBuckets", () => {
  it("keeps incomplete runs live and completed runs in history", () => {
    const buckets = splitRunBuckets([run("training-now", false), run("lenet-previous", true)]);

    expect(buckets.active.map((item) => item.run_id)).toEqual(["training-now"]);
    expect(buckets.history.map((item) => item.run_id)).toEqual(["lenet-previous"]);
  });
});
