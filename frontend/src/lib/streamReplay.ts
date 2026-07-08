import type { RunEvent } from "../api/client";

export function replayDelayMs(index: number, runId?: string): number {
  void index;
  void runId;
  return 0;
}

export function shouldCloseAfterReplay(event: RunEvent): boolean {
  return event.type === "run_complete";
}
