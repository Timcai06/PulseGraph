import type { RunEvent } from "../api/client";

export const RUN_REPLAY_EVENT_DELAY_MS = 140;

export function replayDelayMs(index: number, runId?: string): number {
  return runId ? index * RUN_REPLAY_EVENT_DELAY_MS : 0;
}

export function shouldCloseAfterReplay(event: RunEvent): boolean {
  return event.type === "run_complete";
}
