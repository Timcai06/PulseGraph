from __future__ import annotations

import io
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Literal

import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from app.api.routes_runs import create_runs_router
from app.events.run_registry import run_registry
from app.events.run_store import RunStore
from app.events.training_stream import demo_training_events, to_sse
from app.inspector.fingerprint import fingerprint_state_dict
from app.inspector.fx_tracer import trace_model_graph
from app.inspector.graph_builder import build_graph_from_tensor_specs
from app.inspector.pt_inspector import inspect_pt_file
from app.inspector.safetensors_inspector import inspect_safetensors_file
from app.inspector.source_analyzer import find_module_classes
from app.resources.contract import ResourceContractError, image_shape_from_sample, load_training_resource, model_input_from_sample
from app.runtime.model_loader import forward_with_model, load_model_from_source, validate_source_against_checkpoint
from app.runtime.task_runtime import TaskRuntimeError, resolve_task_runtime
from app.reports.analyzer import build_run_report
from app.reports.markdown import render_run_report_html, render_run_report_markdown
from app.runtime.demo_mlp import demo_graph, run_demo_forward, sample_digit
from app.runtime.replay import ReplayError, build_run_detail, run_replay_forward
from app.schemas import ImageSample, RunDetail, RunEvent


app = FastAPI(title="PulseGraph API", version="0.1.0")
run_store = RunStore()


@app.on_event("startup")
def restore_persisted_runs() -> None:
    run_registry.load_from_store()


MAX_UPLOAD_BYTES = 100 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_ARTIFACT_UPLOAD_BYTES = 50 * 1024 * 1024
SOURCE_TRAIN_EVENT_INTERVAL_SEC = 0.14
RESOURCE_PREVIEW_SAMPLE_LIMIT = 12

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(create_runs_router(run_registry, run_store))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "product": "PulseGraph"}


@app.get("/api/demo/model")
def get_demo_model():
    return demo_graph()


@app.get("/api/demo/forward")
def get_demo_forward(index: int = 0):
    return run_demo_forward(index)


@app.get("/api/runs/demo/stream")
async def stream_demo_run():
    async def generate():
        async for event in demo_training_events():
            yield to_sse(event)

    return StreamingResponse(generate(), media_type="text/event-stream")


def _persist_registration(run_id: str, event: RunEvent) -> None:
    """Registration events also land as standalone provenance files."""
    if event.type == "source_registered":
        classes = event.payload.get("classes") or []
        source_code = "\n\n".join(str(item.get("source_code", "")) for item in classes if isinstance(item, dict))
        if source_code.strip():
            run_store.save_source(run_id, source_code, str(event.payload.get("entry_class") or "") or None)
    elif event.type == "config_registered" and isinstance(event.payload.get("config"), dict):
        run_store.save_config(run_id, event.payload["config"])
    elif event.type == "graph_registered" and isinstance(event.payload.get("graph"), dict):
        run_store.save_graph(run_id, event.payload["graph"])


@app.post("/api/runs/{run_id}/events")
def ingest_run_events(run_id: str, events: RunEvent | list[RunEvent]):
    batch = events if isinstance(events, list) else [events]
    for event in batch:
        event.run_id = run_id
        if event.type == "graph" and isinstance(event.payload.get("tensors"), list):
            graph = build_graph_from_tensor_specs(event.payload["tensors"])
            event.payload = {"graph": graph.model_dump()}
        _persist_registration(run_id, event)
    run = run_registry.publish(run_id, batch)
    return {"accepted": len(batch), "run_id": run_id, "completed": run.completed}


@app.post("/api/runs/{run_id}/checkpoints")
async def upload_checkpoint(run_id: str, request: Request, step: int, epoch: int | None = None):
    data = await request.body()
    if len(data) > MAX_ARTIFACT_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Checkpoint upload is larger than 50 MB.")
    path = run_store.save_checkpoint_bytes(run_id, step, data, epoch)
    fingerprint = None
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict):
            fingerprint = fingerprint_state_dict(state_dict)
            run_store.record_fingerprint(run_id, path.name, fingerprint)
    except Exception:
        pass
    return {"run_id": run_id, "step": step, "path": str(path), "fingerprint": fingerprint}


@app.post("/api/runs/{run_id}/samples")
async def upload_samples(run_id: str, request: Request):
    data = await request.body()
    if len(data) > MAX_ARTIFACT_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Samples upload is larger than 50 MB.")
    path = run_store.save_samples(run_id, data)
    return {"run_id": run_id, "path": str(path)}


async def _collect_source_files(files: list[UploadFile]) -> list[tuple[str, str]]:
    """Decode uploaded .py files (expanding .zip archives) into (relative_path, content)."""
    collected: list[tuple[str, str]] = []
    for upload in files:
        name = (upload.filename or "").replace("\\", "/")
        data = await upload.read()
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    for member in archive.namelist():
                        normalized = member.replace("\\", "/")
                        if not normalized.lower().endswith(".py") or ".." in normalized.split("/"):
                            continue
                        collected.append((normalized, archive.read(member).decode("utf-8", errors="replace")))
            except zipfile.BadZipFile:
                continue
        elif name.lower().endswith(".py"):
            collected.append((name, data.decode("utf-8", errors="replace")))
    return collected


RunEventType = Literal[
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
RunEventSource = Literal["training", "runtime_hook", "checkpoint", "infra", "plugin", "animation"]


def _run_event(
    run_id: str,
    event_type: RunEventType,
    source: RunEventSource,
    step: int,
    payload: dict,
    layer: str | None = None,
) -> RunEvent:
    return RunEvent(
        event_id=str(uuid.uuid4()),
        ts_ns=time.time_ns(),
        source=source,
        type=event_type,
        run_id=run_id,
        step=step,
        layer=layer,
        payload=payload,
    )


def _find_example_input(model: torch.nn.Module) -> torch.Tensor | None:
    for shape in ([1, 1, 28, 28], [1, 3, 32, 32], [1, 784], [1, 10]):
        candidate = torch.rand(*shape)
        try:
            with torch.no_grad():
                output = model(candidate)
            if isinstance(output, torch.Tensor) and output.dim() == 2 and output.shape[0] == 1:
                return candidate
        except Exception:
            continue
    return None


def _publish_source_import_events(run_id: str, graph, layers: list[dict], checkpoint_path: Path, fingerprint: str) -> None:
    events = [
        _run_event(run_id, "graph", "runtime_hook", 0, {"graph": graph.model_dump()}),
    ]
    for layer in layers:
        layer_id = layer.get("layer_id")
        if isinstance(layer_id, str):
            events.append(
                _run_event(
                    run_id,
                    "layer_snapshot",
                    "runtime_hook",
                    1,
                    {
                        "activation_mean": layer.get("activation_mean"),
                        "activation_sparsity": layer.get("activation_sparsity"),
                        "weight_std": layer.get("weight_std"),
                    },
                    layer=layer_id,
                )
            )
    events.extend(
        [
            _run_event(
                run_id,
                "checkpoint",
                "checkpoint",
                1,
                {
                    "path": str(checkpoint_path),
                    "size_mb": round(checkpoint_path.stat().st_size / (1024**2), 3),
                    "fingerprint": fingerprint,
                },
            ),
            _run_event(
                run_id,
                "run_complete",
                "runtime_hook",
                1,
                {"status": "source_imported", "run_kind": "source-import", "inference_only": True},
            ),
        ]
    )
    run_registry.publish(run_id, events)


def _batch_like(example_input: torch.Tensor, batch_size: int) -> torch.Tensor:
    shape = list(example_input.shape)
    shape[0] = batch_size
    return torch.rand(*shape)


def _is_mnist_like(example_input: torch.Tensor) -> bool:
    return list(example_input.shape[1:]) == [1, 28, 28]


def _probe_batch_like(example_input: torch.Tensor, batch_size: int, classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    if _is_mnist_like(example_input):
        images: list[torch.Tensor] = []
        labels: list[int] = []
        visible_classes = max(1, min(classes, 10))
        for index in range(batch_size):
            image, label, _ = sample_digit(index % visible_classes)
            images.append(image)
            labels.append(label if label < classes else label % classes)
        return torch.cat(images, dim=0).to(dtype=example_input.dtype), torch.tensor(labels, dtype=torch.long)
    return _batch_like(example_input, batch_size), torch.randint(0, classes, (batch_size,))


def _class_count(model: torch.nn.Module, example_input: torch.Tensor) -> int | None:
    try:
        with torch.no_grad():
            output = model(example_input)
    except Exception:
        return None
    if isinstance(output, torch.Tensor) and output.dim() == 2 and output.shape[1] > 1:
        return int(output.shape[1])
    return None


def _publish_source_training_events(
    run_id: str,
    graph,
    metrics: list[dict],
    layers_by_step: list[tuple[int, list[dict]]],
    checkpoint_path: Path,
    fingerprint: str,
    run_kind: str = "source-training",
) -> None:
    run_registry.publish(run_id, [_run_event(run_id, "graph", "training", 0, {"graph": graph.model_dump()})])
    if SOURCE_TRAIN_EVENT_INTERVAL_SEC:
        time.sleep(SOURCE_TRAIN_EVENT_INTERVAL_SEC)

    layers_lookup = {step: layers for step, layers in layers_by_step}
    for metric in metrics:
        step = int(metric["step"])
        events = [
            _run_event(
                run_id,
                "metric",
                "training",
                step,
                {
                    "loss": metric["loss"],
                    "accuracy": metric["accuracy"],
                    "learning_rate": metric["learning_rate"],
                    "phase": "train",
                },
            ),
            _run_event(
                run_id,
                "infra",
                "infra",
                step,
                {
                    "device": "cpu",
                    "step_time_ms": metric["step_time_ms"],
                    "samples_per_sec": metric["samples_per_sec"],
                    "memory_peak_mb": 0.0,
                    "elapsed_sec": metric["elapsed_sec"],
                },
            ),
        ]
        for layer in layers_lookup.get(step, []):
            layer_id = layer.get("layer_id")
            if isinstance(layer_id, str):
                events.append(
                    _run_event(
                        run_id,
                        "layer_snapshot",
                        "runtime_hook",
                        step,
                        {
                            "activation_mean": layer.get("activation_mean"),
                            "activation_sparsity": layer.get("activation_sparsity"),
                            "weight_std": layer.get("weight_std"),
                        },
                        layer=layer_id,
                    )
                )
        run_registry.publish(run_id, events)
        if SOURCE_TRAIN_EVENT_INTERVAL_SEC:
            time.sleep(SOURCE_TRAIN_EVENT_INTERVAL_SEC)

    final_step = int(metrics[-1]["step"]) if metrics else 0
    run_registry.publish(
        run_id,
        [
            _run_event(
                run_id,
                "checkpoint",
                "checkpoint",
                final_step,
                {
                    "path": str(checkpoint_path),
                    "size_mb": round(checkpoint_path.stat().st_size / (1024**2), 3),
                    "fingerprint": fingerprint,
                },
            ),
            _run_event(
                run_id,
                "run_complete",
                "training",
                final_step,
                {"status": "trained", "run_kind": run_kind, "inference_only": False},
            ),
        ],
    )


def _run_source_training_job(
    run_id: str,
    model: torch.nn.Module,
    example_input: torch.Tensor,
    classes: int,
    graph,
    steps: int,
) -> None:
    batch_size = 8
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics: list[dict] = []
    layers_by_step: list[tuple[int, list[dict]]] = []
    start = time.perf_counter()

    model.train()
    for step in range(1, steps + 1):
        images, labels = _probe_batch_like(example_input, batch_size, classes)
        step_start = time.perf_counter()
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_time_ms = (time.perf_counter() - step_start) * 1000
        accuracy = float((logits.argmax(dim=1) == labels).float().mean().item())
        metrics.append(
            {
                "step": step,
                "loss": round(float(loss.item()), 4),
                "accuracy": round(accuracy, 4),
                "learning_rate": learning_rate,
                "step_time_ms": round(step_time_ms, 2),
                "samples_per_sec": round(batch_size / max(step_time_ms / 1000, 1e-6), 1),
                "elapsed_sec": round(time.perf_counter() - start, 2),
            }
        )
        model.eval()
        snapshot = forward_with_model(model, images[:1])
        layers_by_step.append((step, snapshot["layers"]))
        model.train()

    run_store.save_config(
        run_id,
        {
            "source": "uploaded-python",
            "run_kind": "source-training",
            "task": "classification",
            "inference_only": False,
            "training_status": "completed",
            "training_recipe": "local-short-classifier",
            "steps": steps,
            "batch_size": batch_size,
            "lr": learning_rate,
            "entry_file": run_store.load_entry_meta(run_id)["entry_file"] if run_store.load_entry_meta(run_id) else None,
            "entry_class": run_store.load_entry_class(run_id),
            "input_shape": list(example_input.shape),
            "classes": classes,
            "weights": "trained",
        },
    )

    sample = io.BytesIO()
    probe_images, probe_labels = _probe_batch_like(example_input, min(16, batch_size), classes)
    torch.save({"images": probe_images.detach().cpu(), "labels": probe_labels.detach().cpu()}, sample)
    run_store.save_samples(run_id, sample.getvalue())

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint_path = run_store.save_checkpoint_bytes(run_id, steps, checkpoint.getvalue(), epoch=1)
    fingerprint = fingerprint_state_dict(model.state_dict())
    run_store.record_fingerprint(run_id, checkpoint_path.name, fingerprint)

    _publish_source_training_events(run_id, graph, metrics, layers_by_step, checkpoint_path, fingerprint)


def _as_model_input(sample: torch.Tensor) -> torch.Tensor:
    return model_input_from_sample(sample)


def _class_name(class_names: list[str] | None, label: int) -> str | None:
    if class_names is None or label < 0 or label >= len(class_names):
        return None
    return class_names[label]


def _resource_info(resource) -> dict:
    return {
        "name": resource.name,
        "task": resource.task,
        "dataset_spec": resource.dataset_spec,
        "output_schema": resource.output_schema,
        "metric_schema": resource.metric_schema,
        "input_shape": resource.input_shape,
        "classes": resource.classes,
        "class_names": resource.class_names,
        "data_source": resource.metadata.get("data_source"),
        "sample_source": resource.sample_source,
    }


def _resource_preview_samples(resource, limit: int = RESOURCE_PREVIEW_SAMPLE_LIMIT) -> list[ImageSample]:
    samples: list[ImageSample] = []
    class_names = resource.class_names
    for index in range(limit):
        image, target = resource.inference_sample(index)
        image_shape = image_shape_from_sample(image, resource.input_shape)
        display_image = image[0] if image.dim() == 4 and image.shape[0] == 1 else image
        label = int(target) if resource.task == "classification" else None
        samples.append(
            ImageSample(
                index=index,
                task=resource.task,
                output=resource.sample_output(target),
                label=label,
                label_name=_class_name(class_names, label) if label is not None else None,
                sample_source=resource.sample_source if resource.sample_source in {"mnist", "synthetic", "probe"} else "probe",
                image_shape=image_shape,
                image_pixels=[float(value) for value in display_image.flatten().tolist()],
            )
        )
    return samples


def _resource_probe_samples(resource, limit: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for index in range(limit):
        image, label = resource.inference_sample(index)
        images.append(_as_model_input(image))
        labels.append(int(label))
    return torch.cat(images, dim=0).detach().cpu(), torch.tensor(labels, dtype=torch.long), resource.sample_source


def _run_resource_training_job(
    run_id: str,
    source_path: Path,
    source_root: Path,
    graph,
    steps: int,
    telemetry_stride: int,
) -> None:
    try:
        resource = load_training_resource(source_path, source_root=source_root)
    except ResourceContractError as exc:
        run_registry.publish(
            run_id,
            [
                _run_event(
                    run_id,
                    "run_complete",
                    "training",
                    0,
                    {"status": "failed", "run_kind": "resource-training", "error": str(exc)},
                )
            ],
        )
        return

    model = resource.build_model()
    runtime = resolve_task_runtime(resource.task)
    runtime.ensure_training_supported()
    batch_size = max(1, min(resource.batch_size, 64))
    learning_rate = resource.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    metrics: list[dict] = []
    layers_by_step: list[tuple[int, list[dict]]] = []
    start = time.perf_counter()

    model.train()
    for step in range(1, steps + 1):
        images, labels = resource.train_batch(step, batch_size)
        step_start = time.perf_counter()
        step_result = runtime.training_step(model, images, labels, optimizer)
        step_time_ms = (time.perf_counter() - step_start) * 1000
        should_record = step % telemetry_stride == 0 or step == steps
        if should_record:
            metrics.append(
                {
                    "step": step,
                    "loss": round(step_result.loss, 4),
                    **{name: round(value, 4) for name, value in step_result.metrics.items()},
                    "learning_rate": learning_rate,
                    "step_time_ms": round(step_time_ms, 2),
                    "samples_per_sec": round(batch_size / max(step_time_ms / 1000, 1e-6), 1),
                    "elapsed_sec": round(time.perf_counter() - start, 2),
                }
            )
            model.eval()
            layers_by_step.append((step, runtime.capture_layers(model, images[:1])))
            model.train()

    run_store.save_config(
        run_id,
        {
            "source": "training-resource",
            "run_kind": "resource-training",
            "resource_name": resource.name,
            "task": resource.task,
            "inference_only": False,
            "training_status": "completed",
            "training_recipe": "resource-contract",
            "steps": steps,
            "telemetry_stride": telemetry_stride,
            "batch_size": batch_size,
            "lr": learning_rate,
            "entry_file": run_store.load_entry_meta(run_id)["entry_file"] if run_store.load_entry_meta(run_id) else None,
            "entry_class": run_store.load_entry_class(run_id),
            "input_shape": resource.input_shape,
            "classes": resource.classes,
            "class_names": resource.class_names,
            "dataset_spec": resource.dataset_spec,
            "output_schema": resource.output_schema,
            "metric_schema": resource.metric_schema,
            "data_source": resource.metadata.get("data_source"),
            "sample_source": resource.sample_source,
            "weights": "trained",
        },
    )

    sample = io.BytesIO()
    probe_images, probe_labels, sample_source = _resource_probe_samples(resource, min(64, batch_size))
    torch.save({"images": probe_images, "labels": probe_labels, "sample_source": sample_source}, sample)
    run_store.save_samples(run_id, sample.getvalue())

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint_path = run_store.save_checkpoint_bytes(run_id, steps, checkpoint.getvalue(), epoch=1)
    fingerprint = fingerprint_state_dict(model.state_dict())
    run_store.record_fingerprint(run_id, checkpoint_path.name, fingerprint)

    _publish_source_training_events(
        run_id,
        graph,
        metrics,
        layers_by_step,
        checkpoint_path,
        fingerprint,
        run_kind="resource-training",
    )


@app.post("/api/runs/train-resource")
async def train_run_from_resource(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    entry_file: str = Form(...),
    steps: int = Form(100),
    telemetry_stride: int = Form(1),
):
    collected = await _collect_source_files(files)
    if not collected:
        raise HTTPException(status_code=400, detail="No Python resource files were found in the upload.")

    run_id = f"resource-{uuid.uuid4().hex[:12]}"
    normalized_entry = _resolve_entry_file(entry_file, collected)
    saved = run_store.save_source_files(run_id, collected, normalized_entry, "TrainingResource")
    if normalized_entry not in saved:
        raise HTTPException(status_code=400, detail=f"Entry file '{entry_file}' was not part of the accepted upload.")

    source_path = run_store.source_path(run_id)
    if source_path is None:
        raise HTTPException(status_code=400, detail="Entry resource could not be saved.")
    source_root = run_store.run_dir(run_id) / "source"
    try:
        resource = load_training_resource(source_path, source_root=source_root)
        resource.ensure_training_supported()
        model = resource.build_model()
        example_input = _as_model_input(resource.inference_sample(0)[0])
    except ResourceContractError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TaskRuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load the training resource: {exc}") from exc

    try:
        graph = trace_model_graph(model, example_input)
    except Exception:
        graph = build_graph_from_tensor_specs(
            [{"name": name, "shape": list(tensor.shape)} for name, tensor in sorted(model.state_dict().items())]
        )

    steps = max(1, min(int(steps), 500))
    telemetry_stride = max(1, min(int(telemetry_stride), steps))
    run_store.save_config(
        run_id,
        {
            "source": "training-resource",
            "run_kind": "resource-training",
            "resource_name": resource.name,
            "task": resource.task,
            "inference_only": False,
            "training_status": "running",
            "training_recipe": "resource-contract",
            "steps": steps,
            "telemetry_stride": telemetry_stride,
            "batch_size": resource.batch_size,
            "lr": resource.learning_rate,
            "entry_file": normalized_entry,
            "entry_class": "TrainingResource",
            "input_shape": resource.input_shape,
            "classes": resource.classes,
            "class_names": resource.class_names,
            "dataset_spec": resource.dataset_spec,
            "output_schema": resource.output_schema,
            "metric_schema": resource.metric_schema,
            "data_source": resource.metadata.get("data_source"),
            "sample_source": resource.sample_source,
            "weights": "training",
        },
    )
    run_store.save_graph(run_id, graph.model_dump())
    run_registry.publish(run_id, [_run_event(run_id, "graph", "training", 0, {"graph": graph.model_dump()})])
    background_tasks.add_task(_run_resource_training_job, run_id, source_path, source_root, graph, steps, telemetry_stride)
    return {
        "run_id": run_id,
        "run_kind": "resource-training",
        "resource": _resource_info(resource),
        "inference_only": False,
        "status": "started",
        "saved": saved,
        "entry_file": normalized_entry,
        "entry_class": "TrainingResource",
        "graph": graph.model_dump(),
        "checkpoint": None,
    }


@app.post("/api/inspect/source/candidates")
async def analyze_source_candidates(files: list[UploadFile] = File(...)):
    collected = await _collect_source_files(files)
    if not collected:
        raise HTTPException(status_code=400, detail="No Python source files were found in the upload.")
    return {"files": [path for path, _ in collected], "candidates": find_module_classes(dict(collected))}


def _resolve_entry_file(requested: str, collected: list[tuple[str, str]]) -> str:
    """Map the requested entry to a collected file. A .zip upload names the
    archive rather than a file inside it, so fall back to conventional roots."""
    names = [path for path, _ in collected]
    normalized = requested.replace("\\", "/")
    if normalized in names:
        return normalized
    if "/" not in normalized:
        leaf_matches = [name for name in names if name.rsplit("/", 1)[-1] == normalized]
        if len(leaf_matches) == 1:
            return leaf_matches[0]
    root_candidates = [name for name in names if "/" not in name]
    for preferred in ("resource.py", "main.py", "train.py"):
        if preferred in root_candidates:
            return preferred
    if len(root_candidates) == 1:
        return root_candidates[0]
    raise HTTPException(
        status_code=400,
        detail=(
            f"Could not determine the entry file for '{requested}'. "
            f"Uploaded files: {', '.join(names[:8])}{'…' if len(names) > 8 else ''}. "
            "Name the entry file resource.py or place a single .py at the archive root."
        ),
    )


@app.post("/api/inspect/resource/preview")
async def preview_training_resource(files: list[UploadFile] = File(...), entry_file: str = Form(...)):
    """Load and fx-trace an uploaded training resource without creating a run,
    so the UI can show the operator graph immediately on import."""
    collected = await _collect_source_files(files)
    if not collected:
        raise HTTPException(status_code=400, detail="No Python resource files were found in the upload.")
    normalized_entry = _resolve_entry_file(entry_file, collected)

    with tempfile.TemporaryDirectory(prefix="pulsegraph-preview-") as tmp:
        root = Path(tmp)
        for rel_path, content in collected:
            target = root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        try:
            resource = load_training_resource(root / normalized_entry, source_root=root)
            model = resource.build_model()
            example_input = _as_model_input(resource.inference_sample(0)[0])
            samples = _resource_preview_samples(resource)
        except ResourceContractError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to load the training resource: {exc}") from exc
        try:
            graph = trace_model_graph(model, example_input)
        except Exception:
            graph = build_graph_from_tensor_specs(
                [{"name": name, "shape": list(tensor.shape)} for name, tensor in sorted(model.state_dict().items())]
            )

    return {
        "resource": _resource_info(resource),
        "samples": [sample.model_dump() for sample in samples],
        "files": [path for path, _ in collected],
        "entry_file": normalized_entry,
        "graph": graph.model_dump(),
    }


@app.post("/api/runs/from-source")
async def import_run_from_source(
    files: list[UploadFile] = File(...),
    entry_file: str = Form(...),
    entry_class: str = Form(...),
):
    """Create a replayable local run from trusted source without requiring trained weights.

    This is intentionally a local/trusted path: the uploaded source is imported
    to instantiate the nn.Module, then PulseGraph saves its initial state_dict as
    a checkpoint so the existing replay, stream, and report surfaces all work.
    """
    collected = await _collect_source_files(files)
    if not collected:
        raise HTTPException(status_code=400, detail="No Python source files were found in the upload.")

    run_id = f"source-{uuid.uuid4().hex[:12]}"
    normalized_entry = entry_file.replace("\\", "/")
    saved = run_store.save_source_files(run_id, collected, normalized_entry, entry_class)
    if normalized_entry not in saved:
        raise HTTPException(status_code=400, detail=f"Entry file '{entry_file}' was not part of the accepted upload.")

    source_path = run_store.source_path(run_id)
    if source_path is None:
        raise HTTPException(status_code=400, detail="Entry source could not be saved.")
    model = load_model_from_source(source_path, entry_class, source_root=run_store.run_dir(run_id) / "source")
    if model is None:
        raise HTTPException(status_code=400, detail=f"Could not instantiate '{entry_class}' from the uploaded source.")
    model.eval()

    example_input = _find_example_input(model)
    if example_input is None:
        raise HTTPException(
            status_code=400,
            detail="Could not infer a runnable input shape. Supported automatic probes include MNIST-like and small RGB classifiers.",
        )

    try:
        forward = forward_with_model(model, example_input)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Forward probe failed: {exc}") from exc

    try:
        graph = trace_model_graph(model, example_input)
    except Exception:
        graph = build_graph_from_tensor_specs(
            [{"name": name, "shape": list(tensor.shape)} for name, tensor in sorted(model.state_dict().items())]
        )

    run_store.save_config(
        run_id,
        {
            "source": "uploaded-python",
            "run_kind": "source-import",
            "task": "classification",
            "inference_only": True,
            "training_status": "not-run",
            "capabilities": ["forward", "stream-replay", "graph", "layer-snapshots"],
            "entry_file": normalized_entry,
            "entry_class": entry_class,
            "input_shape": list(example_input.shape),
            "weights": "initial-random",
        },
    )
    run_store.save_graph(run_id, graph.model_dump())

    sample = io.BytesIO()
    torch.save({"images": example_input.detach().cpu(), "labels": torch.tensor([0])}, sample)
    run_store.save_samples(run_id, sample.getvalue())

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint_path = run_store.save_checkpoint_bytes(run_id, 1, checkpoint.getvalue(), epoch=0)
    fingerprint = fingerprint_state_dict(model.state_dict())
    run_store.record_fingerprint(run_id, checkpoint_path.name, fingerprint)

    _publish_source_import_events(
        run_id,
        graph,
        forward["layers"],
        checkpoint_path=checkpoint_path,
        fingerprint=fingerprint,
    )
    return {
        "run_id": run_id,
        "run_kind": "source-import",
        "inference_only": True,
        "saved": saved,
        "entry_file": normalized_entry,
        "entry_class": entry_class,
        "graph": graph.model_dump(),
        "checkpoint": {"step": 1, "path": str(checkpoint_path), "fingerprint": fingerprint},
    }


@app.post("/api/runs/train-source")
async def train_run_from_source(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    entry_file: str = Form(...),
    entry_class: str = Form(...),
    steps: int = Form(8),
):
    collected = await _collect_source_files(files)
    if not collected:
        raise HTTPException(status_code=400, detail="No Python source files were found in the upload.")

    run_id = f"train-{uuid.uuid4().hex[:12]}"
    normalized_entry = entry_file.replace("\\", "/")
    saved = run_store.save_source_files(run_id, collected, normalized_entry, entry_class)
    if normalized_entry not in saved:
        raise HTTPException(status_code=400, detail=f"Entry file '{entry_file}' was not part of the accepted upload.")

    source_path = run_store.source_path(run_id)
    if source_path is None:
        raise HTTPException(status_code=400, detail="Entry source could not be saved.")
    model = load_model_from_source(source_path, entry_class, source_root=run_store.run_dir(run_id) / "source")
    if model is None:
        raise HTTPException(status_code=400, detail=f"Could not instantiate '{entry_class}' from the uploaded source.")

    example_input = _find_example_input(model)
    if example_input is None:
        raise HTTPException(status_code=400, detail="Could not infer a runnable input shape for training.")
    classes = _class_count(model, example_input)
    if classes is None:
        raise HTTPException(status_code=400, detail="Training requires a classifier forward output shaped [batch, classes].")

    try:
        graph = trace_model_graph(model, example_input)
    except Exception:
        graph = build_graph_from_tensor_specs(
            [{"name": name, "shape": list(tensor.shape)} for name, tensor in sorted(model.state_dict().items())]
        )

    steps = max(1, min(int(steps), 50))
    run_store.save_config(
        run_id,
        {
            "source": "uploaded-python",
            "run_kind": "source-training",
            "task": "classification",
            "inference_only": False,
            "training_status": "running",
            "training_recipe": "local-short-classifier",
            "steps": steps,
            "batch_size": 8,
            "lr": 1e-3,
            "entry_file": normalized_entry,
            "entry_class": entry_class,
            "input_shape": list(example_input.shape),
            "classes": classes,
            "weights": "training",
        },
    )
    run_store.save_graph(run_id, graph.model_dump())
    background_tasks.add_task(_run_source_training_job, run_id, model, example_input, classes, graph, steps)
    return {
        "run_id": run_id,
        "run_kind": "source-training",
        "inference_only": False,
        "status": "started",
        "saved": saved,
        "entry_file": normalized_entry,
        "entry_class": entry_class,
        "graph": graph.model_dump(),
        "checkpoint": None,
    }


@app.post("/api/runs/import")
async def import_artifact(request: Request):
    """Create a synthetic run for a bare .pt so source can be attached and it becomes replayable."""
    data = await request.body()
    if len(data) > MAX_ARTIFACT_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Artifact is larger than 50 MB.")
    try:
        state_dict = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        assert isinstance(state_dict, dict) and state_dict
        fingerprint = fingerprint_state_dict(state_dict)
    except Exception:
        raise HTTPException(status_code=400, detail="Artifact is not a loadable weights-only state dict.")
    existing = run_store.find_run_by_fingerprint(fingerprint)
    if existing:
        return {"run_id": existing[0], "fingerprint": fingerprint, "created": False}
    run_id = f"imported-{fingerprint[:12]}"
    path = run_store.save_checkpoint_bytes(run_id, 0, data)
    run_store.record_fingerprint(run_id, path.name, fingerprint)
    return {"run_id": run_id, "fingerprint": fingerprint, "created": True}


@app.post("/api/runs/{run_id}/source")
async def attach_run_source(
    run_id: str,
    files: list[UploadFile] = File(...),
    entry_file: str = Form(...),
    entry_class: str = Form(...),
):
    collected = await _collect_source_files(files)
    if not collected:
        raise HTTPException(status_code=400, detail="No Python source files were found in the upload.")
    saved = run_store.save_source_files(run_id, collected, entry_file.replace("\\", "/"), entry_class)
    if entry_file not in saved:
        raise HTTPException(status_code=400, detail=f"Entry file '{entry_file}' was not part of the accepted upload.")

    validation = None
    checkpoint_path = run_store.checkpoint_path(run_id)
    if checkpoint_path is not None:
        source_path = run_store.source_path(run_id)
        if source_path is not None:
            validation = validate_source_against_checkpoint(
                source_path, entry_class, checkpoint_path, source_root=run_store.run_dir(run_id) / "source"
            ).as_dict()
    return {"run_id": run_id, "saved": saved, "entry_file": entry_file, "entry_class": entry_class, "validation": validation}


@app.get("/api/runs/{run_id}/detail")
def get_run_detail(run_id: str) -> RunDetail:
    detail = build_run_detail(run_store, run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return detail


@app.get("/api/runs/{run_id}/source")
def get_run_source(run_id: str) -> PlainTextResponse:
    source = run_store.load_source(run_id)
    if source is None:
        raise HTTPException(status_code=404, detail="This run has no recorded model source.")
    return PlainTextResponse(source)


@app.get("/api/runs/{run_id}/graph")
def get_run_graph(run_id: str):
    graph = run_store.load_graph(run_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="This run has no recorded graph.")
    return graph


@app.get("/api/runs/{run_id}/checkpoints")
def list_run_checkpoints(run_id: str):
    return run_store.list_checkpoints(run_id)


@app.get("/api/runs/{run_id}/forward")
def replay_run_forward(run_id: str, checkpoint_step: int = 0, index: int = 0):
    try:
        return run_replay_forward(run_store, run_id, checkpoint_step, index)
    except ReplayError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error


@app.get("/api/runs/{run_id}/report")
def get_run_report(run_id: str):
    detail = build_run_detail(run_store, run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return build_run_report(run_store, detail)


@app.get("/api/runs/{run_id}/report/export.md")
def export_run_report_markdown(run_id: str):
    detail = build_run_detail(run_store, run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    report = build_run_report(run_store, detail)
    markdown = render_run_report_markdown(detail, report)
    return Response(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{run_id}-report.md"'},
    )


@app.get("/api/runs/{run_id}/report/export.html")
def export_run_report_html(run_id: str):
    detail = build_run_detail(run_store, run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    report = build_run_report(run_store, detail)
    html = render_run_report_html(detail, report)
    return Response(content=html, media_type="text/html; charset=utf-8")


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str):
    async def generate():
        async for event in run_registry.subscribe(run_id):
            if event is None:
                yield ": keepalive\n\n"
            else:
                yield to_sse(event)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/inspect/upload")
async def inspect_upload(file: UploadFile = File(...)):
    filename = file.filename or "model.pt"
    suffix = Path(filename).suffix or ".pt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
        total = 0
        while chunk := await file.read(UPLOAD_CHUNK_BYTES):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Uploaded model file is larger than 100 MB.")
            handle.write(chunk)
        handle.flush()
        if suffix.lower() == ".safetensors":
            response = inspect_safetensors_file(Path(handle.name), display_filename=filename)
        else:
            response = inspect_pt_file(Path(handle.name), display_filename=filename)
        if response.weights_fingerprint:
            match = run_store.find_run_by_fingerprint(response.weights_fingerprint)
            if match:
                run_id, checkpoint_step = match
                response.matched_run_id = run_id
                response.warnings.insert(
                    0,
                    f"Weights match recorded run '{run_id}' (checkpoint step {checkpoint_step}); "
                    "full training provenance is available.",
                )
        return response.model_dump()
