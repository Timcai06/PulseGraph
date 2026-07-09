import type { GraphNode, LayerSnapshot } from "../api/client";

export type HealthSeverity = "healthy" | "caution" | "warning" | "unknown";

export type LayerHealthSummary = {
  severity: HealthSeverity;
  label: string;
  detail: string;
};

function formatShape(shape?: number[] | null) {
  return shape?.length ? `[${shape.join("x")}]` : "?";
}

export function formatNodeShape(node: GraphNode, snapshot?: LayerSnapshot) {
  const input = snapshot?.input_shape ?? node.input_shape;
  const output = snapshot?.output_shape ?? node.output_shape;
  return `${formatShape(input)} -> ${formatShape(output)}`;
}

export function formatParamCount(value: number) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

export function deriveLayerHealth(node: GraphNode, snapshot?: LayerSnapshot): LayerHealthSummary {
  if (!snapshot) {
    return {
      severity: node.confidence === "inferred" ? "caution" : "unknown",
      label: node.confidence,
      detail: `Graph confidence is ${node.confidence}; no live snapshot recorded.`
    };
  }

  if (snapshot.activation_sparsity != null && snapshot.activation_sparsity >= 0.95) {
    return {
      severity: "warning",
      label: "sparse activation",
      detail: "Activation sparsity is high; possible dead layer."
    };
  }

  if (snapshot.gradient_norm != null && snapshot.gradient_norm <= 1e-7) {
    return {
      severity: "warning",
      label: "low gradient",
      detail: "Gradient norm is near zero; possible vanishing gradient."
    };
  }

  if (snapshot.gradient_norm != null && snapshot.gradient_norm >= 100) {
    return {
      severity: "warning",
      label: "high gradient",
      detail: "Gradient norm is very high; possible exploding gradient."
    };
  }

  if (snapshot.activation_sparsity != null && snapshot.activation_sparsity >= 0.8) {
    return {
      severity: "caution",
      label: "activation caution",
      detail: "Activation sparsity is elevated."
    };
  }

  return {
    severity: "healthy",
    label: "healthy",
    detail: "Latest layer snapshot is within the local Ops thresholds."
  };
}
