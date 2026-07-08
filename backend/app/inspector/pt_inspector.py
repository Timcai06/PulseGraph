from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import torch

from app.inspector.artifact_registry import register_artifact
from app.inspector.fingerprint import fingerprint_state_dict
from app.inspector.graph_builder import build_inferred_graph
from app.schemas import InspectionResponse, TensorSummary


CHECKPOINT_STATE_KEYS = (
    "state_dict",
    "model_state_dict",
    "model",
    "module",
)


def summarize_tensor(name: str, tensor: torch.Tensor) -> TensorSummary:
    detached = tensor.detach().float().cpu()
    return TensorSummary(
        name=name,
        shape=list(tensor.shape),
        dtype=str(tensor.dtype).replace("torch.", ""),
        numel=tensor.numel(),
        mean=float(detached.mean().item()) if detached.numel() else None,
        std=float(detached.std(unbiased=False).item()) if detached.numel() else None,
        min=float(detached.min().item()) if detached.numel() else None,
        max=float(detached.max().item()) if detached.numel() else None,
    )


def _extract_state_dict(obj: Any) -> tuple[str, dict[str, torch.Tensor], list[str]]:
    warnings: list[str] = []
    if isinstance(obj, dict) and all(isinstance(value, torch.Tensor) for value in obj.values()):
        return "state_dict", obj, warnings

    if isinstance(obj, dict):
        for key in CHECKPOINT_STATE_KEYS:
            value = obj.get(key)
            if isinstance(value, dict) and all(isinstance(item, torch.Tensor) for item in value.values()):
                warnings.append(f"Loaded checkpoint bundle and selected '{key}' as the state dict.")
                return "checkpoint", value, warnings

    warnings.append("The file did not contain a plain tensor state_dict or recognized checkpoint bundle.")
    return "unknown", {}, warnings


def inspect_loaded_object(filename: str, obj: Any) -> InspectionResponse:
    mode, state_dict, warnings = _extract_state_dict(obj)
    tensors = [summarize_tensor(name, tensor) for name, tensor in sorted(state_dict.items())]
    graph = build_inferred_graph(tensors)
    if mode != "unknown":
        warnings.append("Graph is inferred from parameter tensors; non-parameter layers may be missing.")
    return InspectionResponse(
        filename=filename,
        mode=mode,
        safe=True,
        weights_fingerprint=fingerprint_state_dict(state_dict) if state_dict else None,
        tensors=tensors,
        graph=graph,
        warnings=warnings,
    )


def _looks_like_torchscript(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return any(name.endswith("constants.pkl") for name in archive.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


def _inspect_torchscript(filename: str, path: Path) -> InspectionResponse | None:
    """Inspect a TorchScript archive via torch.jit.load (restricted unpickler, no forward run)."""
    if not _looks_like_torchscript(path):
        return None
    try:
        module = torch.jit.load(str(path), map_location="cpu")
        state_dict = dict(module.state_dict())
    except Exception:
        return None
    tensors = [summarize_tensor(name, tensor) for name, tensor in sorted(state_dict.items())]
    return InspectionResponse(
        filename=filename,
        mode="torchscript",
        safe=True,
        weights_fingerprint=fingerprint_state_dict(state_dict) if state_dict else None,
        tensors=tensors,
        graph=build_inferred_graph(tensors),
        warnings=[
            "Loaded as a TorchScript archive with torch.jit.load; no forward pass was executed.",
            "Graph is inferred from parameter tensors; non-parameter layers may be missing.",
        ],
    )


def inspect_pt_file(path: Path, display_filename: str | None = None) -> InspectionResponse:
    artifact = register_artifact(path)
    filename = display_filename or path.name
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        response = _inspect_torchscript(filename, path)
        if response is None:
            response = InspectionResponse(
                filename=filename,
                mode="unknown",
                safe=True,
                tensors=[],
                graph=build_inferred_graph([]),
                warnings=[
                    "Safe weights-only loading failed.",
                    "The artifact could not be decoded as tensor weights without unsafe deserialization.",
                ],
            )
        response.artifact_id = artifact.artifact_id
        response.artifact_sha256 = artifact.sha256
        return response
    response = inspect_loaded_object(filename, obj)
    response.artifact_id = artifact.artifact_id
    response.artifact_sha256 = artifact.sha256
    return response
