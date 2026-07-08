from __future__ import annotations

from pathlib import Path

from safetensors import safe_open

from app.inspector.artifact_registry import register_artifact
from app.inspector.fingerprint import fingerprint_state_dict
from app.inspector.graph_builder import build_inferred_graph
from app.inspector.pt_inspector import summarize_tensor
from app.schemas import InspectionResponse


def inspect_safetensors_file(path: Path, display_filename: str | None = None) -> InspectionResponse:
    """Safe inspection of a .safetensors file; the format cannot embed code."""
    artifact = register_artifact(path)
    filename = display_filename or path.name
    try:
        tensors = []
        state_dict = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in sorted(handle.keys()):
                tensor = handle.get_tensor(name)
                state_dict[name] = tensor
                tensors.append(summarize_tensor(name, tensor))
    except Exception:
        return InspectionResponse(
            artifact_id=artifact.artifact_id,
            artifact_sha256=artifact.sha256,
            filename=filename,
            mode="unknown",
            safe=True,
            tensors=[],
            graph=build_inferred_graph([]),
            warnings=["The file could not be parsed as a safetensors archive."],
        )
    graph = build_inferred_graph(tensors)
    return InspectionResponse(
        artifact_id=artifact.artifact_id,
        artifact_sha256=artifact.sha256,
        filename=filename,
        mode="safetensors",
        safe=True,
        weights_fingerprint=fingerprint_state_dict(state_dict) if state_dict else None,
        tensors=tensors,
        graph=graph,
        warnings=["Graph is inferred from parameter tensors; non-parameter layers may be missing."],
    )
