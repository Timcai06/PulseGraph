from __future__ import annotations

import math
import re
from typing import Any

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


def build_graph_from_tensor_specs(specs: list[dict[str, Any]]) -> ModelGraph:
    """Build an inferred graph from lightweight {name, shape} specs (e.g. a live run's graph event)."""
    tensors = [
        TensorSummary(
            name=str(spec["name"]),
            shape=[int(dim) for dim in spec.get("shape", [])],
            dtype=str(spec.get("dtype", "unknown")),
            numel=math.prod(int(dim) for dim in spec.get("shape", [])) if spec.get("shape") else 0,
        )
        for spec in specs
        if isinstance(spec, dict) and spec.get("name")
    ]
    return build_inferred_graph(tensors)


def build_bounded_graph_from_tensor_specs(specs: list[dict[str, Any]], max_nodes: int = 24) -> ModelGraph:
    tensors = [
        TensorSummary(
            name=str(spec["name"]),
            shape=[int(dim) for dim in spec.get("shape", [])],
            dtype=str(spec.get("dtype", "unknown")),
            numel=math.prod(int(dim) for dim in spec.get("shape", [])) if spec.get("shape") else 0,
        )
        for spec in specs
        if isinstance(spec, dict) and spec.get("name")
    ]
    return build_bounded_inferred_graph(tensors, max_nodes=max_nodes)


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


def build_bounded_inferred_graph(tensors: list[TensorSummary], max_nodes: int = 24) -> ModelGraph:
    by_layer: dict[str, list[TensorSummary]] = {}
    for tensor in tensors:
        key = _layer_key(tensor.name)
        by_layer.setdefault(key, []).append(tensor)

    weighted_layers = [
        (layer_key, values)
        for layer_key, values in by_layer.items()
        if any(item.name.endswith(".weight") for item in values)
    ]
    weighted_layers.sort(key=lambda item: _natural_sort_key(item[0]))
    group_budget = max(1, max_nodes - 2)
    if len(weighted_layers) <= group_budget:
        return build_inferred_graph(tensors)

    layer_names = [layer_key for layer_key, _ in weighted_layers]
    chosen_depth = 1
    chosen_groups: dict[str, list[str]] = {}
    for depth in range(1, 5):
        groups: dict[str, list[str]] = {}
        for layer_name in layer_names:
            parts = layer_name.split(".")
            prefix = ".".join(parts[: min(depth, len(parts))])
            groups.setdefault(prefix, []).append(layer_name)
        if len(groups) <= group_budget:
            chosen_depth = depth
            chosen_groups = groups
    if not chosen_groups:
        chunk_size = math.ceil(len(layer_names) / group_budget)
        chosen_groups = {
            f"group_{index // chunk_size + 1:02d}": layer_names[index : index + chunk_size]
            for index in range(0, len(layer_names), chunk_size)
        }
        chosen_depth = 0

    nodes: list[GraphNode] = [
        GraphNode(
            id="input",
            label="Input",
            kind="Input",
            confidence="inferred",
            metadata={"note": "Input is inferred from grouped parameter summaries."},
        )
    ]
    edges: list[GraphEdge] = []
    previous = "input"
    for prefix in sorted(chosen_groups, key=_natural_sort_key):
        group_layers = set(chosen_groups[prefix])
        group_tensors = [tensor for tensor in tensors if _layer_key(tensor.name) in group_layers]
        param_count = sum(tensor.numel for tensor in group_tensors)
        sample_layers = sorted(group_layers, key=_natural_sort_key)[:4]
        metadata: dict[str, Any] = {
            "aggregation": "bounded-module-group",
            "aggregation_depth": chosen_depth,
            "weighted_layers": len(group_layers),
            "tensor_count": len(group_tensors),
            "sample_layers": sample_layers,
        }
        if len(group_layers) > len(sample_layers):
            metadata["remaining_layers"] = len(group_layers) - len(sample_layers)
        nodes.append(
            GraphNode(
                id=prefix,
                label=prefix,
                kind="ModuleGroup",
                param_count=param_count,
                confidence="inferred",
                metadata=metadata,
            )
        )
        edges.append(GraphEdge(id=f"{previous}->{prefix}", source=previous, target=prefix))
        previous = prefix

    nodes.append(
        GraphNode(
            id="output",
            label="Output",
            kind="Output",
            confidence="inferred",
            metadata={
                "note": "Grouped graph is a bounded summary because the parameter graph exceeded the node budget.",
            },
        )
    )
    edges.append(GraphEdge(id=f"{previous}->output", source=previous, target="output"))
    return ModelGraph(nodes=nodes, edges=edges)
