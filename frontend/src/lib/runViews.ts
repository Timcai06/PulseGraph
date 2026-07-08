import type { RunSummary } from "../api/client";

export type RunBuckets = {
  active: RunSummary[];
  history: RunSummary[];
};

export function splitRunBuckets(runs: RunSummary[]): RunBuckets {
  return {
    active: runs.filter((run) => !run.completed),
    history: runs.filter((run) => run.completed)
  };
}
