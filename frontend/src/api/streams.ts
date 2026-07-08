import type { RunEvent, RunEventType } from "./types";

const EVENT_TYPES: RunEventType[] = [
  "metric",
  "layer_snapshot",
  "infra",
  "checkpoint",
  "animation",
  "graph",
  "source_registered",
  "config_registered",
  "graph_registered",
  "run_complete"
];

function openStream(url: string, onEvent: (event: RunEvent) => void): EventSource {
  const source = new EventSource(url);
  for (const type of EVENT_TYPES) {
    source.addEventListener(type, (message) => {
      onEvent(JSON.parse((message as MessageEvent).data) as RunEvent);
    });
  }
  return source;
}

export function openDemoStream(onEvent: (event: RunEvent) => void): EventSource {
  return openStream("/api/runs/demo/stream", onEvent);
}

export function openRunStream(runId: string, onEvent: (event: RunEvent) => void): EventSource {
  return openStream(`/api/runs/${encodeURIComponent(runId)}/stream`, onEvent);
}
