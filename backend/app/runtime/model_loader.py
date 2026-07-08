from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_model_from_source(source_path: Path, entry_class: str, source_root: Path | None = None) -> nn.Module | None:
    """Dynamically load a model class from recorded source and instantiate it.

    Executes the source file, so it must only ever be called on source stored in
    the local runs/ directory (recorded at training time or explicitly attached
    by the user). Multi-file uploads work via `source_root` on sys.path, so the
    entry file can use absolute imports relative to the upload root
    (e.g. `from modules.attention import Block`).
    """
    module_name = f"pulsegraph_run_model_{uuid.uuid4().hex[:8]}"
    root = str(source_root or source_path.parent)
    sys.path.insert(0, root)
    try:
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        model_class = getattr(module, entry_class, None)
        if model_class is None or not issubclass(model_class, nn.Module):
            return None
        return model_class()
    except Exception:
        return None
    finally:
        sys.modules.pop(module_name, None)
        if root in sys.path:
            sys.path.remove(root)


class SourceValidation:
    def __init__(self, ok: bool, error: str | None = None, missing_keys: list[str] | None = None, unexpected_keys: list[str] | None = None) -> None:
        self.ok = ok
        self.error = error
        self.missing_keys = missing_keys or []
        self.unexpected_keys = unexpected_keys or []

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "missing_keys": self.missing_keys[:20],
            "unexpected_keys": self.unexpected_keys[:20],
        }


def validate_source_against_checkpoint(
    source_path: Path, entry_class: str, checkpoint_path: Path, source_root: Path | None = None
) -> SourceValidation:
    """Dry-run: rebuild the model and check the checkpoint fits it, with precise feedback."""
    model = load_model_from_source(source_path, entry_class, source_root)
    if model is None:
        return SourceValidation(False, error=f"Could not instantiate '{entry_class}' from the uploaded source (import or constructor failed).")
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        return SourceValidation(False, error="Checkpoint could not be loaded as a weights-only state dict.")
    if not isinstance(state_dict, dict):
        return SourceValidation(False, error="Checkpoint does not contain a state dict.")
    try:
        result = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        # strict=False still raises on shape mismatches
        return SourceValidation(False, error=f"Source and weights do not match: {exc}")
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        return SourceValidation(False, error="Source and weights do not match.", missing_keys=missing, unexpected_keys=unexpected)
    return SourceValidation(True)


def load_model_and_weights(source_path: Path, entry_class: str, checkpoint_path: Path, source_root: Path | None = None) -> nn.Module | None:
    model = load_model_from_source(source_path, entry_class, source_root)
    if model is None:
        return None
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    except Exception:
        return None
    model.eval()
    return model


def forward_with_model(model: nn.Module, image_tensor: torch.Tensor) -> dict[str, Any]:
    """Run one forward pass on any model, capturing per-layer activations via hooks."""
    layers: list[dict[str, Any]] = []
    handles = []

    def make_hook(layer_name: str):
        def hook(_module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor):
                return
            detached = output.detach()
            input_shape = list(inputs[0].shape[1:]) if inputs and isinstance(inputs[0], torch.Tensor) else None
            layers.append(
                {
                    "layer_id": layer_name,
                    "input_shape": input_shape,
                    "output_shape": list(detached.shape[1:]) if detached.dim() > 1 else list(detached.shape),
                    "activation_mean": round(float(detached.float().mean()), 4),
                    "activation_sparsity": round(float((detached == 0).float().mean()), 4),
                }
            )

        return hook

    for name, module in model.named_modules():
        if name and any(True for _ in module.parameters(recurse=False)):
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            logits = model(image_tensor)
    finally:
        for handle in handles:
            handle.remove()

    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    return {
        "logits": [round(float(value), 4) for value in logits.squeeze(0).tolist()],
        "probabilities": [float(value) for value in probabilities.tolist()],
        "prediction": int(probabilities.argmax().item()),
        "layers": layers,
    }
