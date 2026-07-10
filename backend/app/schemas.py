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
    weights_fingerprint: str | None = None
    matched_run_id: str | None = None
    filename: str
    mode: Literal["state_dict", "checkpoint", "safetensors", "torchscript", "unknown"]
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
    task: str = "classification"
    output: dict[str, Any] = Field(default_factory=dict)
    sample_index: int
    label: int | None = None
    prediction: int | None = None
    weights: Literal["trained", "random"] = "random"
    sample_source: Literal["mnist", "synthetic", "probe"] = "synthetic"
    class_names: list[str] | None = None
    image_shape: list[int] = Field(default_factory=list)
    image_pixels: list[float]
    probabilities: list[float] = Field(default_factory=list)
    graph: ModelGraph
    layers: list[LayerSnapshot]


class ImageSample(BaseModel):
    index: int
    task: str = "classification"
    output: dict[str, Any] = Field(default_factory=dict)
    label: int | None = None
    label_name: str | None = None
    sample_source: Literal["mnist", "synthetic", "probe"] = "probe"
    image_shape: list[int]
    image_pixels: list[float]


class CheckpointInfo(BaseModel):
    step: int
    epoch: int | None = None
    path: str
    size_mb: float
    fingerprint: str | None = None


class RunDetail(BaseModel):
    """Aggregated provenance for one run: source, config, graph, metrics, checkpoints."""

    run_id: str
    created_at: float
    completed: bool
    source: str | None = None
    entry_class: str | None = None
    source_files: list[str] = Field(default_factory=list)
    source_origin: Literal["recorded", "user-attached"] | None = None
    config: dict[str, Any] | None = None
    graph: ModelGraph | None = None
    has_samples: bool = False
    metrics: list[dict[str, Any]] = Field(default_factory=list)
    checkpoints: list[CheckpointInfo] = Field(default_factory=list)
    event_count: int = 0


class RunInsight(BaseModel):
    severity: Literal["info", "warning", "critical"]
    title: str
    detail: str
    suggestion: str | None = None


class LayerHealth(BaseModel):
    layer_id: str
    mean_sparsity: float | None = None
    last_gradient_norm: float | None = None
    gradient_trend: Literal["stable", "vanishing", "exploding", "unknown"] = "unknown"
    weight_std_drift: float | None = None


class CheckpointEvaluation(BaseModel):
    step: int
    accuracy: float | None = None
    sample_count: int = 0


class ErrorAnalysis(BaseModel):
    confusion: list[list[int]] = Field(default_factory=list)
    labels: list[int] = Field(default_factory=list)
    class_names: list[str] | None = None
    misclassified: list[dict[str, Any]] = Field(default_factory=list)


class RunReport(BaseModel):
    run_id: str
    generated_for_checkpoint: int | None = None
    final_loss: float | None = None
    best_accuracy: float | None = None
    overfit_gap: float | None = None
    loss_plateau_step: int | None = None
    layer_health: list[LayerHealth] = Field(default_factory=list)
    checkpoint_evaluations: list[CheckpointEvaluation] = Field(default_factory=list)
    error_analysis: ErrorAnalysis | None = None
    insights: list[RunInsight] = Field(default_factory=list)


class RunSummary(BaseModel):
    run_id: str
    created_at: float
    last_event_at: float
    completed: bool
    event_count: int
    last_step: int


class RunEvent(BaseModel):
    event_id: str
    schema_version: str = "pulsegraph.event.v1"
    ts_ns: int
    source: Literal["training", "runtime_hook", "checkpoint", "infra", "plugin", "animation"]
    type: Literal[
        "metric",
        "layer_snapshot",
        "infra",
        "checkpoint",
        "animation",
        "graph",
        "source_registered",
        "config_registered",
        "graph_registered",
        "run_complete",
    ]
    run_id: str
    session_id: str | None = None
    step: int
    epoch: int | None = None
    layer: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
