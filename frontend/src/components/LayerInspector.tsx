import type { GraphNode, InspectionResponse, LayerSnapshot, TensorSummary } from "../api/client";

function fmtShape(shape?: number[] | null) {
  return shape && shape.length ? shape.join(" x ") : "unknown";
}

type Props = {
  node?: GraphNode;
  tensors: TensorSummary[];
  layer?: LayerSnapshot;
  inspection?: InspectionResponse;
};

export function LayerInspector({ node, tensors, layer, inspection }: Props) {
  if (!node) {
    return (
      <aside className="inspector">
        <h2>Layer Inspector</h2>
        <p className="hint">Select a layer in the graph to inspect shape, parameters, and live layer signals.</p>
      </aside>
    );
  }

  const related = tensors.filter((tensor) => tensor.name.startsWith(node.id));

  return (
    <aside className="inspector">
      <h2>{node.label}</h2>
      <div className="pill">{node.kind}</div>
      {inspection && (
        <section className="artifact-card">
          <h3>{inspection.filename}</h3>
          <span>{inspection.mode} · {inspection.safe ? "safe weights-only" : "unsafe"}</span>
          {inspection.artifact_id && <code>{inspection.artifact_id.slice(0, 18)}</code>}
          {inspection.artifact_sha256 && <span>sha256 {inspection.artifact_sha256.slice(0, 16)}</span>}
        </section>
      )}
      <dl>
        <dt>Input</dt><dd>{fmtShape(node.input_shape)}</dd>
        <dt>Output</dt><dd>{fmtShape(node.output_shape)}</dd>
        <dt>Params</dt><dd>{node.param_count.toLocaleString()}</dd>
        <dt>Confidence</dt><dd>{node.confidence}</dd>
      </dl>

      {layer && (
        <section className="signal-card">
          <h3>Live Layer Pulse</h3>
          <dl>
            <dt>Activation mean</dt><dd>{layer.activation_mean?.toFixed(4) ?? "n/a"}</dd>
            <dt>Sparsity</dt><dd>{layer.activation_sparsity?.toFixed(4) ?? "n/a"}</dd>
            <dt>Gradient norm</dt><dd>{layer.gradient_norm?.toFixed(4) ?? "n/a"}</dd>
            <dt>Weight std</dt><dd>{layer.weight_std?.toFixed(4) ?? "n/a"}</dd>
          </dl>
        </section>
      )}

      {related.length > 0 && (
        <section>
          <h3>Parameter tensors</h3>
          {related.map((tensor) => (
            <div className="tensor-row" key={tensor.name}>
              <strong>{tensor.name}</strong>
              <span>{fmtShape(tensor.shape)} · {tensor.dtype}</span>
              <span>mean {tensor.mean?.toFixed(4)} · std {tensor.std?.toFixed(4)}</span>
            </div>
          ))}
        </section>
      )}
    </aside>
  );
}
