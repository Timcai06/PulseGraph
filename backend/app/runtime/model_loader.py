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


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def forward_with_layer_capture(model: nn.Module, *args: Any, **kwargs: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Run an inference call while collecting tensor activations from parameterized layers."""
    layers: list[dict[str, Any]] = []
    handles = []

    def make_hook(layer_name: str):
        def hook(module: nn.Module, inputs: tuple, output: Any) -> None:
            detached = _first_tensor(output)
            if detached is None:
                return
            detached = detached.detach()
            input_tensor = _first_tensor(inputs)
            input_shape = None
            if input_tensor is not None:
                input_shape = list(input_tensor.shape[1:]) if input_tensor.dim() >= 4 else list(input_tensor.shape)
            # 参数侧统计：weight_std 直接读权重；gradient_norm 读 param.grad ——
            # 训练循环在 optimizer.step() 之后、下一轮 zero_grad 之前采样，
            # 此时梯度仍是本步的值。纯推理路径没有梯度，该项保持 None。
            weight = getattr(module, "weight", None)
            weight_std = None
            if isinstance(weight, torch.Tensor) and weight.numel() > 1:
                weight_std = round(float(weight.detach().float().std()), 6)
            grad_sq = 0.0
            has_grad = False
            for param in module.parameters(recurse=False):
                if param.grad is not None:
                    grad_sq += float(param.grad.detach().float().pow(2).sum())
                    has_grad = True
            layers.append(
                {
                    "layer_id": layer_name,
                    "input_shape": input_shape,
                    "output_shape": list(detached.shape[1:]) if detached.dim() > 1 else list(detached.shape),
                    "activation_mean": round(float(detached.float().mean()), 4),
                    "activation_sparsity": round(float((detached == 0).float().mean()), 4),
                    "weight_std": weight_std,
                    "gradient_norm": round(grad_sq**0.5, 6) if has_grad else None,
                }
            )

        return hook

    for name, module in model.named_modules():
        if name and any(True for _ in module.parameters(recurse=False)):
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            output = model(*args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()

    return output, layers


def forward_with_model(model: nn.Module, image_tensor: torch.Tensor) -> dict[str, Any]:
    """Run a classification forward pass and capture per-layer activations."""
    logits, layers = forward_with_layer_capture(model, image_tensor)
    if not isinstance(logits, torch.Tensor) or logits.dim() != 2:
        raise ValueError("Classification model output must have shape [batch, classes].")

    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    return {
        "logits": [round(float(value), 4) for value in logits.squeeze(0).tolist()],
        "probabilities": [float(value) for value in probabilities.tolist()],
        "prediction": int(probabilities.argmax().item()),
        "layers": layers,
    }
