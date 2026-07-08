from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TensorSummary(BaseModel):
    name: str
    shape: list[int]
    dtype: str
    numel: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None


class GraphNode(BaseModel):
    id: str
    label: str
    kind: str
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    param_count: int = 0
    confidence: Literal["safe", "inferred", "trusted"] = "inferred"
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str | None = None


class ModelGraph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class InspectionResponse(BaseModel):
    artifact_id: str | None = None
    artifact_sha256: str | None = None
    filename: str
    mode: Literal["state_dict", "checkpoint", "unknown"]
    safe: bool
    tensors: list[TensorSummary]
    graph: ModelGraph
    warnings: list[str] = Field(default_factory=list)


class ArtifactRecord(BaseModel):
    artifact_id: str
    path: str
    filename: str
    sha256: str
    size_bytes: int
    format: str
    trust_level: Literal["untrusted", "trusted"] = "untrusted"


class LayerSnapshot(BaseModel):
    layer_id: str
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    activation_mean: float | None = None
    activation_sparsity: float | None = None
    gradient_norm: float | None = None
    weight_std: float | None = None


class PredictionResponse(BaseModel):
    sample_index: int
    label: int
    prediction: int
    probabilities: list[float]
    graph: ModelGraph
    layers: list[LayerSnapshot]


class RunEvent(BaseModel):
    event_id: str
    schema_version: str = "pulsegraph.event.v1"
    ts_ns: int
    source: Literal["training", "runtime_hook", "checkpoint", "infra", "plugin", "animation"]
    type: Literal["metric", "layer_snapshot", "infra", "checkpoint", "animation", "run_complete"]
    run_id: str
    session_id: str | None = None
    step: int
    epoch: int | None = None
    layer: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
