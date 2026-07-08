from __future__ import annotations

import re

from app.schemas import GraphEdge, GraphNode, ModelGraph, TensorSummary


def _layer_key(tensor_name: str) -> str:
    if tensor_name.endswith(".weight"):
        return tensor_name[: -len(".weight")]
    if tensor_name.endswith(".bias"):
        return tensor_name[: -len(".bias")]
    return tensor_name


def _natural_sort_key(value: str) -> list[int | str]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts]


def build_inferred_graph(tensors: list[TensorSummary]) -> ModelGraph:
    by_layer: dict[str, dict[str, TensorSummary]] = {}
    for tensor in tensors:
        key = _layer_key(tensor.name)
        by_layer.setdefault(key, {})[tensor.name] = tensor

    nodes: list[GraphNode] = [
        GraphNode(
            id="input",
            label="Input",
            kind="Input",
            confidence="inferred",
            metadata={"note": "Input is inferred from the first parameterized layer."},
        )
    ]

    weighted_layers = [
        (layer_key, values)
        for layer_key, values in by_layer.items()
        if any(name.endswith(".weight") for name in values)
    ]
    weighted_layers.sort(key=lambda item: _natural_sort_key(item[0]))

    previous = "input"
    edges: list[GraphEdge] = []

    for layer_key, values in weighted_layers:
        weight = next(tensor for name, tensor in values.items() if name.endswith(".weight"))
        bias = next((tensor for name, tensor in values.items() if name.endswith(".bias")), None)
        kind = "Linear" if len(weight.shape) == 2 else "Conv2d" if len(weight.shape) == 4 else "Parameter"
        input_shape = None
        output_shape = None
        metadata: dict[str, object] = {}

        if kind == "Linear":
            output_features, input_features = weight.shape
            input_shape = [input_features]
            output_shape = [output_features]
            metadata = {"input_features": input_features, "output_features": output_features}
        elif kind == "Conv2d":
            out_channels, in_channels, kernel_h, kernel_w = weight.shape
            input_shape = [in_channels, kernel_h, kernel_w]
            output_shape = [out_channels]
            metadata = {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": [kernel_h, kernel_w],
            }

        param_count = weight.numel + (bias.numel if bias else 0)
        node = GraphNode(
            id=layer_key,
            label=layer_key,
            kind=kind,
            input_shape=input_shape,
            output_shape=output_shape,
            param_count=param_count,
            confidence="inferred",
            metadata=metadata,
        )
        nodes.append(node)
        edges.append(GraphEdge(id=f"{previous}->{layer_key}", source=previous, target=layer_key))
        previous = layer_key

    if len(nodes) == 1:
        nodes[0].metadata["note"] = "No weighted layers could be inferred from tensor names."

    return ModelGraph(nodes=nodes, edges=edges)

