import { useCallback, useEffect, useReducer, useRef } from "react";
import { openDemoStream, openRunStream, type LayerSnapshot, type RunEvent } from "../api/client";

export type StreamStatus = "idle" | "streaming" | "complete" | "error";

export type MetricPoint = {
  step: number;
  loss?: number;
  accuracy?: number;
  stepTimeMs?: number;
  memoryPeakMb?: number;
};

type StreamState = {
  status: StreamStatus;
  runId?: string;
  metrics: MetricPoint[];
  events: RunEvent[];
  layerSnapshots: Record<string, LayerSnapshot>;
  pulsedNodeId?: string;
  device: string;
};

type StreamAction =
  | { type: "reset" }
  | { type: "start"; runId?: string }
  | { type: "status"; status: StreamStatus }
  | { type: "event"; event: RunEvent }
  | { type: "snapshots"; snapshots: Record<string, LayerSnapshot>; pulsedNodeId?: string };

const MAX_EVENTS = 60;

const initialState: StreamState = {
  status: "idle",
  metrics: [],
  events: [],
  layerSnapshots: {},
  device: "unknown"
};

function upsertMetric(metrics: MetricPoint[], step: number, patch: Partial<MetricPoint>): MetricPoint[] {
  for (let index = metrics.length - 1; index >= 0; index -= 1) {
    if (metrics[index].step === step) {
      const next = metrics.slice();
      next[index] = { ...next[index], ...patch };
      return next;
    }
    if (metrics[index].step < step) break;
  }
  return [...metrics, { step, ...patch }];
}

function applyEvent(state: StreamState, event: RunEvent): StreamState {
  if (state.events.some((existing) => existing.event_id === event.event_id)) return state;
  const next: StreamState = { ...state, events: [event, ...state.events].slice(0, MAX_EVENTS) };

  switch (event.type) {
    case "metric": {
      const patch: Partial<MetricPoint> = {};
      if (event.payload.loss != null) patch.loss = event.payload.loss;
      if (event.payload.accuracy != null) patch.accuracy = event.payload.accuracy;
      next.metrics = upsertMetric(state.metrics, event.step, patch);
      return next;
    }
    case "infra": {
      const patch: Partial<MetricPoint> = {};
      if (event.payload.step_time_ms != null) patch.stepTimeMs = event.payload.step_time_ms;
      if (event.payload.memory_peak_mb != null) patch.memoryPeakMb = event.payload.memory_peak_mb;
      next.metrics = upsertMetric(state.metrics, event.step, patch);
      if (event.payload.device) next.device = event.payload.device;
      return next;
    }
    case "layer_snapshot": {
      if (!event.layer) return next;
      next.layerSnapshots = {
        ...state.layerSnapshots,
        [event.layer]: {
          layer_id: event.layer,
          activation_mean: event.payload.activation_mean,
          activation_sparsity: event.payload.activation_sparsity,
          gradient_norm: event.payload.gradient_norm,
          weight_std: event.payload.weight_std
        }
      };
      next.pulsedNodeId = event.layer;
      return next;
    }
    case "animation": {
      const path = event.payload.path ?? undefined;
      next.pulsedNodeId = path?.length ? path[event.step % path.length] : state.pulsedNodeId;
      return next;
    }
    case "run_complete":
      next.status = "complete";
      return next;
    default:
      return next;
  }
}

function reducer(state: StreamState, action: StreamAction): StreamState {
  switch (action.type) {
    case "reset":
      return initialState;
    case "start":
      return { ...initialState, status: "streaming", runId: action.runId };
    case "status":
      return { ...state, status: action.status };
    case "event":
      return applyEvent(state, action.event);
    case "snapshots":
      return { ...state, layerSnapshots: action.snapshots, pulsedNodeId: action.pulsedNodeId };
    default:
      return state;
  }
}

export function useRunStream() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const sourceRef = useRef<EventSource | null>(null);

  const closeSource = useCallback(() => {
    sourceRef.current?.close();
    sourceRef.current = null;
  }, []);

  const startStream = useCallback(
    (runId?: string) => {
      closeSource();
      dispatch({ type: "start", runId });
      const onEvent = (event: RunEvent) => {
        dispatch({ type: "event", event });
        if (event.type === "run_complete") closeSource();
      };
      const source = runId ? openRunStream(runId, onEvent) : openDemoStream(onEvent);
      source.onerror = () => {
        if (source.readyState === EventSource.CLOSED) {
          dispatch({ type: "status", status: "error" });
          closeSource();
        }
      };
      sourceRef.current = source;
    },
    [closeSource]
  );

  const reset = useCallback(() => {
    closeSource();
    dispatch({ type: "reset" });
  }, [closeSource]);

  const applyPrediction = useCallback((layers: LayerSnapshot[], pulsedNodeId?: string) => {
    dispatch({
      type: "snapshots",
      snapshots: Object.fromEntries(layers.map((layer) => [layer.layer_id, layer])),
      pulsedNodeId
    });
  }, []);

  useEffect(() => closeSource, [closeSource]);

  return { ...state, startStream, reset, applyPrediction };
}
