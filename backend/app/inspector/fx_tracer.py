from __future__ import annotations

import torch
from torch import fx, nn
from torch.fx.passes.shape_prop import ShapeProp

from app.schemas import GraphEdge, GraphNode, ModelGraph


def _shape_of(node: fx.Node) -> list[int] | None:
    meta = node.meta.get("tensor_meta")
    if meta is None or not hasattr(meta, "shape"):
        return None
    shape = list(meta.shape)
    return shape[1:] if len(shape) > 1 else shape


def _kind_of(node: fx.Node, modules: dict[str, nn.Module]) -> str:
    if node.op == "placeholder":
        return "Input"
    if node.op == "call_module":
        return type(modules[str(node.target)]).__name__
    if node.op == "call_function":
        return getattr(node.target, "__name__", str(node.target)).capitalize()
    if node.op == "call_method":
        return str(node.target).capitalize()
    return node.op


def _param_count(node: fx.Node, modules: dict[str, nn.Module]) -> int:
    if node.op != "call_module":
        return 0
    return sum(parameter.numel() for parameter in modules[str(node.target)].parameters())


def trace_model_graph(
    model: nn.Module,
    example_input: torch.Tensor,
    rename: dict[str, str] | None = None,
) -> ModelGraph:
    """Build a ModelGraph from a real torch.fx trace with propagated shapes.

    Only call this for trusted, runnable models: symbolic tracing executes the
    module's forward with proxy tensors. `rename` maps fx targets (for module
    calls) or node names to friendly node ids.
    """
    rename = rename or {}
    traced = fx.symbolic_trace(model)
    ShapeProp(traced).propagate(example_input)
    modules = dict(traced.named_modules())

    def node_id(node: fx.Node) -> str:
        if node.op == "call_module":
            target = str(node.target)
            return rename.get(target, target)
        if node.op == "placeholder":
            return rename.get(node.name, "input")
        return rename.get(node.name, node.name)

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    known: dict[fx.Node, str] = {}

    for node in traced.graph.nodes:
        if node.op == "output":
            continue
        identifier = node_id(node)
        known[node] = identifier
        kind = _kind_of(node, modules)
        input_shapes = [_shape_of(arg) for arg in node.all_input_nodes if arg in known]
        nodes.append(
            GraphNode(
                id=identifier,
                label=identifier if node.op == "placeholder" else f"{kind} · {identifier}",
                kind=kind,
                input_shape=next((shape for shape in input_shapes if shape), None),
                output_shape=_shape_of(node),
                param_count=_param_count(node, modules),
                confidence="trusted",
                metadata={"fx_op": node.op},
            )
        )
        for source in node.all_input_nodes:
            if source in known and source is not node:
                edges.append(GraphEdge(id=f"{known[source]}->{identifier}", source=known[source], target=identifier))

    return ModelGraph(nodes=nodes, edges=edges)
