export type ForwardTarget = {
  runId: string;
  checkpointStep?: number;
};

export type ForwardRequest =
  | { mode: "demo" }
  | { mode: "run"; runId: string; checkpointStep: number };

export function resolveForwardTarget(target?: ForwardTarget): ForwardRequest {
  if (!target) return { mode: "demo" };
  return { mode: "run", runId: target.runId, checkpointStep: target.checkpointStep ?? 0 };
}

export function describeForwardTarget(target?: ForwardTarget): string {
  return target ? target.runId : "demo";
}
