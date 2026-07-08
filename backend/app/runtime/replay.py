from __future__ import annotations

import torch

from app.events.run_registry import run_registry
from app.events.run_store import RunStore
from app.inspector.graph_builder import build_inferred_graph
from app.inspector.pt_inspector import summarize_tensor
from app.resources.contract import ResourceContractError, load_training_resource
from app.runtime import mnist_data
from app.runtime.model_loader import forward_with_model, load_model_and_weights
from app.schemas import LayerSnapshot, ModelGraph, PredictionResponse, RunDetail


class ReplayError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def load_probe_samples(store: RunStore, run_id: str) -> tuple[torch.Tensor, torch.Tensor, str] | None:
    """Probe samples recorded at training time: {"images": [N,...], "labels": [N]}."""
    path = store.samples_path(run_id)
    if path is None:
        return None
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        images, labels = bundle["images"], bundle["labels"]
        sample_source = str(bundle.get("sample_source") or "probe")
    except Exception:
        return None
    if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor) or not images.shape[0]:
        return None
    return images.float(), labels.long(), sample_source


def _pick_sample(store: RunStore, run_id: str, index: int) -> tuple[torch.Tensor, int, str]:
    probe = load_probe_samples(store, run_id)
    if probe is not None:
        images, labels, sample_source = probe
        position = index % images.shape[0]
        return images[position : position + 1], int(labels[position].item()), sample_source
    mnist = mnist_data.load_test_samples()
    if mnist is not None:
        images, labels = mnist
        position = index % images.shape[0]
        return images[position : position + 1], int(labels[position].item()), "mnist"
    raise ReplayError(400, "No probe samples were recorded for this run and no MNIST fallback is available.")


def _run_graph(store: RunStore, run_id: str, checkpoint_path) -> ModelGraph:
    recorded = store.load_graph(run_id)
    if recorded:
        try:
            return ModelGraph.model_validate(recorded)
        except Exception:
            pass
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        tensors = [summarize_tensor(name, tensor) for name, tensor in sorted(state_dict.items())]
        return build_inferred_graph(tensors)
    except Exception:
        return build_inferred_graph([])


def run_replay_forward(store: RunStore, run_id: str, checkpoint_step: int = 0, index: int = 0) -> PredictionResponse:
    """Reconstruct a run's model from recorded source + checkpoint and run inference."""
    source_path = store.source_path(run_id)
    entry_class = store.load_entry_class(run_id)
    config = store.load_config(run_id) or {}
    if source_path is None or entry_class is None:
        raise ReplayError(400, "This run has no recorded model source; forward replay is unavailable.")
    checkpoint_path = store.checkpoint_path(run_id, checkpoint_step or None)
    if checkpoint_path is None:
        raise ReplayError(404, "No matching checkpoint was found for this run.")
    if config.get("run_kind") == "resource-training":
        try:
            resource = load_training_resource(source_path, source_root=store.run_dir(run_id) / "source")
            model = resource.build_model()
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(state_dict, dict):
                raise ReplayError(500, "Recorded checkpoint is not a state_dict.")
            model.load_state_dict(state_dict)
            model.eval()
        except ResourceContractError as exc:
            raise ReplayError(400, str(exc)) from exc
        except ReplayError:
            raise
        except Exception as exc:
            raise ReplayError(500, f"Failed to rebuild the resource model: {exc}") from exc
    else:
        model = load_model_and_weights(source_path, entry_class, checkpoint_path, source_root=store.run_dir(run_id) / "source")
        if model is None:
            raise ReplayError(500, "Failed to rebuild the model from recorded source and checkpoint.")

    image, label, sample_source = _pick_sample(store, run_id, index)
    try:
        result = forward_with_model(model, image)
    except Exception as exc:
        raise ReplayError(500, f"Forward pass failed: {exc}") from exc

    return PredictionResponse(
        sample_index=index,
        label=label,
        prediction=result["prediction"],
        weights="random" if config.get("weights") == "initial-random" or config.get("inference_only") is True else "trained",
        sample_source=sample_source if sample_source in {"probe", "mnist", "synthetic"} else "synthetic",
        image_pixels=[float(value) for value in image.flatten().tolist()],
        probabilities=result["probabilities"],
        graph=_run_graph(store, run_id, checkpoint_path),
        layers=[LayerSnapshot(**layer) for layer in result["layers"]],
    )


def build_run_detail(store: RunStore, run_id: str) -> RunDetail | None:
    events = store.load_events(run_id)
    live = run_registry.get(run_id)
    if not events and live is None and store.source_path(run_id) is None:
        return None

    metrics = [
        {"step": event.step, "epoch": event.epoch, **event.payload}
        for event in events
        if event.type == "metric"
    ]
    graph_dict = store.load_graph(run_id)
    graph = None
    if graph_dict:
        try:
            graph = ModelGraph.model_validate(graph_dict)
        except Exception:
            graph = None

    created_at = live.created_at if live else (events[0].ts_ns / 1e9 if events else 0.0)
    completed = live.completed if live else any(event.type == "run_complete" for event in events)

    entry_meta = store.load_entry_meta(run_id)
    return RunDetail(
        run_id=run_id,
        created_at=created_at,
        completed=completed,
        source=store.load_source(run_id),
        entry_class=entry_meta["entry_class"] if entry_meta else None,
        source_files=store.list_source_files(run_id),
        source_origin=entry_meta.get("origin") if entry_meta else None,  # type: ignore[arg-type]
        config=store.load_config(run_id),
        graph=graph,
        has_samples=store.samples_path(run_id) is not None,
        metrics=metrics,
        checkpoints=store.list_checkpoints(run_id),
        event_count=len(events) if events else (live.event_count if live else 0),
    )
