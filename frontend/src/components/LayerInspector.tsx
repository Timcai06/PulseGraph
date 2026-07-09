import { X } from "lucide-react";
import type { GraphNode, LayerSnapshot, RunEvent } from "../api/client";
import type { LayerHistoryPoint } from "../hooks/useRunStream";
import { deriveLayerHealth, formatNodeShape, formatParamCount } from "../lib/layerHealth";

type Props = {
  node?: GraphNode;
  snapshot?: LayerSnapshot;
  history: LayerHistoryPoint[];
  events: RunEvent[];
  selectedStep?: number;
  onClose?: () => void;
};

function value(value?: number | null) {
  return value == null ? "n/a" : Number(value).toPrecision(4);
}

export function LayerInspector({ node, snapshot, history, events, selectedStep, onClose }: Props) {
  if (!node) {
    return (
      <section className="layer-inspector empty">
        <h2>Layer Inspector</h2>
        <p>Select an operator node to inspect runtime health.</p>
      </section>
    );
  }

  const health = deriveLayerHealth(node, snapshot);
  const relatedEvents = events.filter((event) => event.layer === node.id).slice(0, 5);

  return (
    <section className={`layer-inspector health-${health.severity}`}>
      <header>
        <div>
          <span>Layer Inspector</span>
          <h2>{node.label || node.id}</h2>
          <p>
            {node.kind} · {node.confidence}
          </p>
        </div>
        {onClose && (
          <button className="layer-inspector-close" onClick={onClose} type="button" aria-label="Close layer inspector">
            <X size={14} />
          </button>
        )}
      </header>
      <dl>
        <div>
          <dt>shape</dt>
          <dd>{formatNodeShape(node, snapshot)}</dd>
        </div>
        <div>
          <dt>params</dt>
          <dd>{formatParamCount(node.param_count)}</dd>
        </div>
        <div>
          <dt>activation sparsity</dt>
          <dd>{value(snapshot?.activation_sparsity)}</dd>
        </div>
        <div>
          <dt>gradient norm</dt>
          <dd>{value(snapshot?.gradient_norm)}</dd>
        </div>
        <div>
          <dt>weight std</dt>
          <dd>{value(snapshot?.weight_std)}</dd>
        </div>
      </dl>
      <div className="layer-health-note">
        <strong>{health.label}</strong>
        <span>{health.detail}</span>
      </div>
      <div className="layer-history-note">
        {history.length ? `${history.length} snapshots recorded${selectedStep == null ? "" : ` · replay <= step ${selectedStep}`}` : "No history for this layer yet"}
      </div>
      <div className="layer-events">
        {relatedEvents.length ? (
          relatedEvents.map((event) => (
            <span key={event.event_id}>
              {event.type} · step {event.step}
            </span>
          ))
        ) : (
          <span>No related events</span>
        )}
      </div>
    </section>
  );
}
