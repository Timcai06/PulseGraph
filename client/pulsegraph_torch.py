"""PyTorch-side helpers for PulseGraph telemetry (companion to pulsegraph_client).

Keep pulsegraph_client stdlib-only; anything that needs torch lives here.
"""

from __future__ import annotations

import inspect
import io
import textwrap
from typing import Any

import torch
from torch import fx, nn
from torch.fx.passes.shape_prop import ShapeProp


DEFAULT_SOURCE_HEADER = "import torch\nfrom torch import nn\n\n\n"


def extract_model_source(model_class: type) -> dict[str, Any]:
    """Extract the model class source (plus torch imports) for provenance recording.

    Returns {"source_code": str, "entry_class": str}. Raises OSError/TypeError when
    the source is unavailable (e.g. classes defined in a REPL) — callers should
    fall back to saving the whole training script file.
    """
    source = textwrap.dedent(inspect.getsource(model_class))
    return {"source_code": DEFAULT_SOURCE_HEADER + source, "entry_class": model_class.__name__}


def trace_model_graph(model: nn.Module, example_input: torch.Tensor) -> dict[str, Any]:
    """Exact compute graph via torch.fx with propagated shapes, as a ModelGraph dict.

    Only call on your own trusted model: symbolic tracing executes forward with proxies.
    Raises on dynamic control flow — callers should fall back to state_dict_specs().
    """
    traced = fx.symbolic_trace(model)
    ShapeProp(traced).propagate(example_input)
    modules = dict(traced.named_modules())

    def shape_of(node: fx.Node) -> list[int] | None:
        meta = node.meta.get("tensor_meta")
        if meta is None or not hasattr(meta, "shape"):
            return None
        shape = list(meta.shape)
        return shape[1:] if len(shape) > 1 else shape

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    known: dict[fx.Node, str] = {}

    for node in traced.graph.nodes:
        if node.op == "output":
            continue
        if node.op == "placeholder":
            identifier, kind, params = "input", "Input", 0
        elif node.op == "call_module":
            identifier = str(node.target)
            module = modules[identifier]
            kind = type(module).__name__
            params = sum(parameter.numel() for parameter in module.parameters())
        else:
            identifier = node.name
            kind = getattr(node.target, "__name__", str(node.target)).capitalize()
            params = 0
        known[node] = identifier
        input_shapes = [shape_of(arg) for arg in node.all_input_nodes if arg in known]
        nodes.append(
            {
                "id": identifier,
                "label": identifier if kind == "Input" else f"{kind} · {identifier}",
                "kind": kind,
                "input_shape": next((shape for shape in input_shapes if shape), None),
                "output_shape": shape_of(node),
                "param_count": params,
                "confidence": "trusted",
                "metadata": {"fx_op": node.op},
            }
        )
        for source in node.all_input_nodes:
            if source in known and source is not node:
                edges.append({"id": f"{known[source]}->{identifier}", "source": known[source], "target": identifier})

    return {"nodes": nodes, "edges": edges}


def register_samples(run: Any, images: torch.Tensor, labels: torch.Tensor, limit: int = 16) -> bool:
    """Record a small probe batch with the run so replay works for any data domain."""
    count = min(limit, images.shape[0])
    buffer = io.BytesIO()
    torch.save({"images": images[:count].detach().cpu(), "labels": labels[:count].detach().cpu()}, buffer)
    return bool(run.upload_samples(buffer.getvalue()))


def state_dict_specs(model: nn.Module) -> list[dict[str, Any]]:
    """Lightweight {name, shape, dtype} specs so the backend can infer the model graph."""
    return [
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", "")}
        for name, tensor in model.state_dict().items()
    ]


class LayerProbes:
    """Forward hooks on every parameterized leaf module, capturing activation stats.

    Layer names follow module qualified names (e.g. "net.1"), matching the ids of
    the graph the backend infers from state_dict_specs().
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._activations: dict[str, dict[str, float]] = {}
        for name, module in model.named_modules():
            if name and any(True for _ in module.parameters(recurse=False)):
                module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            if isinstance(output, torch.Tensor):
                detached = output.detach()
                self._activations[name] = {
                    "activation_mean": float(detached.mean()),
                    "activation_sparsity": float((detached == 0).float().mean()),
                }

        return hook

    def snapshots(self) -> dict[str, dict[str, float]]:
        """Latest per-layer stats, enriched with weight std and gradient norm."""
        result: dict[str, dict[str, float]] = {}
        for name, module in self._model.named_modules():
            if name not in self._activations:
                continue
            snapshot = dict(self._activations[name])
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor):
                snapshot["weight_std"] = float(weight.detach().std())
                if weight.grad is not None:
                    snapshot["gradient_norm"] = float(weight.grad.detach().norm())
            result[name] = {key: round(value, 4) for key, value in snapshot.items()}
        return result
