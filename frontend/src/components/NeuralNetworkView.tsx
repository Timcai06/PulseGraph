import { useEffect, useMemo, useRef, useState } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import type { ModelGraph } from "../api/client";

gsap.registerPlugin(useGSAP);

type NeuralRole = "input" | "hidden" | "output";

export type NeuralNeuron = {
  id: string;
  label: string;
  intensity: number;
  activationOrder: number;
  strongest?: boolean;
};

export type NeuralLayer = {
  id: string;
  label: string;
  role: NeuralRole;
  sourceNodeId: string;
  active: boolean;
  neurons: NeuralNeuron[];
};

type BuildNeuralLayersInput = {
  graph: ModelGraph;
  probabilities?: number[];
  pulsedNodeId?: string;
};

const MAX_VISIBLE_NEURONS = 10;

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function makeNeurons(prefix: string, count: number, layerIndex: number, intensities?: number[]) {
  const visibleCount = Math.max(1, Math.min(MAX_VISIBLE_NEURONS, count));
  const maxIntensity = intensities ? Math.max(...intensities) : undefined;
  return Array.from({ length: visibleCount }, (_, index) => ({
    id: `${prefix}-${index}`,
    label: String(index),
    intensity: clamp01(intensities?.[index] ?? 0.34 + index / Math.max(visibleCount * 3, 1)),
    activationOrder: layerIndex * MAX_VISIBLE_NEURONS + index,
    strongest: maxIntensity !== undefined && intensities?.[index] === maxIntensity
  }));
}

export function buildNeuralLayers({ graph, probabilities, pulsedNodeId }: BuildNeuralLayersInput): NeuralLayer[] {
  const linearNodes = graph.nodes.filter((node) => node.kind === "Linear");
  const hiddenNode = linearNodes[0] ?? graph.nodes[0];
  const outputNode = graph.nodes.find((node) => node.kind === "Softmax") ?? linearNodes[linearNodes.length - 1] ?? graph.nodes[graph.nodes.length - 1];

  const layers: NeuralLayer[] = [];
  if (hiddenNode) {
    const hiddenWidth = hiddenNode.output_shape?.[0] ?? 8;
    layers.push({
      id: "neural-hidden",
      label: hiddenNode.label,
      role: "hidden",
      sourceNodeId: hiddenNode.id,
      active: pulsedNodeId === hiddenNode.id,
      neurons: makeNeurons("hidden", hiddenWidth, 0)
    });
  }
  if (outputNode) {
    const outputWidth = probabilities?.length || outputNode.output_shape?.[0] || 10;
    layers.push({
      id: "neural-output",
      label: "Digit scores",
      role: "output",
      sourceNodeId: outputNode.id,
      active: pulsedNodeId === outputNode.id,
      neurons: makeNeurons("output", outputWidth, 1, probabilities)
    });
  }

  return layers;
}

type Props = {
  graph: ModelGraph;
  probabilities?: number[];
  pulsedNodeId?: string;
  onSelectLayer: (nodeId: string) => void;
};

export function NeuralNetworkView({ graph, probabilities, pulsedNodeId, onSelectLayer }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [reducedMotion, setReducedMotion] = useState(false);
  const layers = useMemo(() => buildNeuralLayers({ graph, probabilities, pulsedNodeId }), [graph, probabilities, pulsedNodeId]);

  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReducedMotion(query.matches);
    const onChange = () => setReducedMotion(query.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  useGSAP(() => {
    if (!pulsedNodeId || reducedMotion) return;
    const target = containerRef.current?.querySelector(`[data-neural-layer="${pulsedNodeId}"]`);
    const neurons = pulsedNodeId === "softmax"
      ? containerRef.current?.querySelectorAll(".neuron")
      : target?.querySelectorAll(".neuron");
    if (!neurons?.length) return;
    gsap.fromTo(
      neurons,
      { scale: 0.92, opacity: 0.72 },
      {
        scale: 1.12,
        opacity: 1,
        stagger: pulsedNodeId === "softmax" ? 0.035 : 0.03,
        repeat: 1,
        yoyo: true,
        duration: 0.2,
        ease: "power2.out"
      }
    );
  }, { dependencies: [pulsedNodeId, reducedMotion, layers.length], scope: containerRef });

  return (
    <div className="neural-view" ref={containerRef}>
      <svg className="neural-links" aria-hidden="true">
        {layers.slice(0, -1).map((layer, layerIndex) => {
          const nextLayer = layers[layerIndex + 1];
          return layer.neurons.slice(0, 6).flatMap((sourceNeuron, sourceIndex) =>
            nextLayer.neurons.slice(0, 6).map((targetNeuron, targetIndex) => (
              <line
                key={`${sourceNeuron.id}-${targetNeuron.id}`}
                x1={`${18 + layerIndex * 32}%`}
                y1={`${18 + sourceIndex * 11}%`}
                x2={`${18 + (layerIndex + 1) * 32}%`}
                y2={`${18 + targetIndex * 11}%`}
              />
            ))
          );
        })}
      </svg>
      <div className="neural-layers">
        {layers.map((layer) => (
          <button
            className={`neural-layer ${layer.active ? "active" : ""}`}
            data-neural-layer={layer.sourceNodeId}
            key={layer.id}
            onClick={() => onSelectLayer(layer.sourceNodeId)}
            type="button"
          >
            <span className="neural-layer-title">{layer.label}</span>
            <span className="neural-layer-role">{layer.role}</span>
            <span className="neuron-stack">
              {layer.neurons.map((neuron) => (
                <span
                  className="neuron"
                  key={neuron.id}
                  data-neuron-order={neuron.activationOrder}
                  data-strongest={neuron.strongest ? "true" : undefined}
                  style={{ "--intensity": neuron.intensity } as React.CSSProperties}
                >
                  {layer.role === "output" ? neuron.label : ""}
                </span>
              ))}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
