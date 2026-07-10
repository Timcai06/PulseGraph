from __future__ import annotations

import io
import time
from pathlib import Path

import torch

from app.events.run_registry import RunRegistry
from app.events.run_store import RunStore
from app.events.training_control import TrainingTaskController
from app.events.training_events import (
    build_run_event,
    publish_run_status,
    publish_training_progress,
    publish_training_stage,
    save_run_config,
)
from app.inspector.fingerprint import fingerprint_state_dict
from app.resources.contract import load_training_resource, model_input_from_sample
from app.runtime.task_runtime import resolve_task_runtime


def resource_probe_samples(resource, limit: int) -> tuple[torch.Tensor, object, str]:
    images: list[torch.Tensor] = []
    targets: list[object] = []
    for index in range(limit):
        image, target = resource.inference_sample(index)
        images.append(model_input_from_sample(image))
        targets.append(target)
    return (
        torch.cat(images, dim=0).detach().cpu(),
        resource.runtime.pack_probe_targets(targets),
        resource.sample_source,
    )


def _optimizer_learning_rate(optimizer: torch.optim.Optimizer, fallback: float) -> float:
    if not optimizer.param_groups:
        return fallback
    value = optimizer.param_groups[0].get("lr", fallback)
    return float(value) if isinstance(value, (int, float)) else fallback


def _set_training_phase(
    run_store: RunStore,
    run_registry: RunRegistry,
    run_id: str,
    phase: str,
    message: str,
    *,
    steps: int,
    progress: float,
) -> None:
    save_run_config(run_store, run_id, training_status=phase)
    publish_run_status(run_registry, run_id, phase, message, total_steps=steps, progress=progress)


def _start_stage(
    run_registry: RunRegistry,
    run_id: str,
    scope: str,
    stage: str,
    message: str,
    *,
    step: int,
    steps: int,
    progress: float | None = None,
) -> float:
    publish_training_stage(
        run_registry,
        run_id,
        scope,
        stage,
        "active",
        message,
        step=step,
        total_steps=steps,
        progress=progress,
    )
    return time.perf_counter()


def _finish_stage(
    run_registry: RunRegistry,
    run_id: str,
    scope: str,
    stage: str,
    message: str,
    started_at: float,
    *,
    step: int,
    steps: int,
    progress: float | None = None,
) -> None:
    publish_training_stage(
        run_registry,
        run_id,
        scope,
        stage,
        "completed",
        message,
        step=step,
        total_steps=steps,
        duration_ms=(time.perf_counter() - started_at) * 1000,
        progress=progress,
    )


def run_resource_training_job(
    run_id: str,
    source_path: Path,
    source_root: Path,
    steps: int,
    telemetry_stride: int,
    *,
    run_store: RunStore,
    run_registry: RunRegistry,
    training_task_controller: TrainingTaskController,
    max_live_layer_samples: int,
) -> None:
    training_task_controller.register(run_id)
    acquired = training_task_controller.acquire(run_id)
    if acquired.completed and acquired.phase == "cancelled":
        save_run_config(
            run_store,
            run_id,
            training_status="cancelled",
            cancel_requested=True,
            completed_at=time.time(),
            last_step=0,
            weights="training",
        )
        publish_run_status(
            run_registry,
            run_id,
            "cancelled",
            "Training cancelled before execution started.",
            total_steps=steps,
            progress=0.0,
        )
        run_registry.publish(
            run_id,
            [build_run_event(run_id, "run_complete", "training", 0, {"status": "cancelled", "run_kind": "resource-training"})],
        )
        return

    last_step = 0
    start = time.perf_counter()
    active_stage = "queued"
    try:
        publish_training_stage(
            run_registry,
            run_id,
            "lifecycle",
            "queued",
            "completed",
            "Training left the queue.",
            total_steps=steps,
            progress=0.01,
        )
        save_run_config(run_store, run_id, training_status="loading", started_at=time.time(), cancel_requested=False)
        publish_run_status(run_registry, run_id, "loading", "Loading resource package.", total_steps=steps, progress=0.01)
        active_stage = "loading"
        stage_started = _start_stage(
            run_registry, run_id, "lifecycle", "loading", "Loading resource package.", step=0, steps=steps, progress=0.01
        )
        resource = load_training_resource(source_path, source_root=source_root)
        runtime = resolve_task_runtime(resource.task)
        runtime.ensure_training_supported()
        _finish_stage(
            run_registry, run_id, "lifecycle", "loading", "Resource package loaded.", stage_started, step=0, steps=steps, progress=0.03
        )

        if training_task_controller.is_cancel_requested(run_id):
            raise RuntimeError("__cancelled__")

        _set_training_phase(
            run_store, run_registry, run_id, "building", "Building and validating model.", steps=steps, progress=0.03
        )
        active_stage = "building"
        stage_started = _start_stage(
            run_registry, run_id, "lifecycle", "building", "Building and validating model.", step=0, steps=steps, progress=0.03
        )
        model = resource.build_model()
        _finish_stage(
            run_registry, run_id, "lifecycle", "building", "Model ready.", stage_started, step=0, steps=steps, progress=0.05
        )
        batch_size = max(1, min(resource.batch_size, 64))
        _set_training_phase(
            run_store, run_registry, run_id, "preparing_data", "Preparing telemetry probe samples.", steps=steps, progress=0.05
        )
        active_stage = "preparing_data"
        stage_started = _start_stage(
            run_registry, run_id, "lifecycle", "preparing_data", "Preparing telemetry probe samples.", step=0, steps=steps, progress=0.05
        )
        probe_images, probe_targets, sample_source = resource_probe_samples(resource, min(64, batch_size))
        _finish_stage(
            run_registry, run_id, "lifecycle", "preparing_data", "Probe samples ready.", stage_started, step=0, steps=steps, progress=0.07
        )
        _set_training_phase(
            run_store, run_registry, run_id, "initializing", "Initializing optimizer and training hooks.", steps=steps, progress=0.07
        )
        active_stage = "initializing"
        stage_started = _start_stage(
            run_registry, run_id, "lifecycle", "initializing", "Initializing optimizer and training hooks.", step=0, steps=steps, progress=0.07
        )
        optimizer = resource.build_optimizer(model)
        learning_rate = _optimizer_learning_rate(optimizer, resource.learning_rate)
        _finish_stage(
            run_registry, run_id, "lifecycle", "initializing", "Optimizer and hooks ready.", stage_started, step=0, steps=steps, progress=0.08
        )

        model.train()
        save_run_config(
            run_store,
            run_id,
            training_status="running",
            batch_size=batch_size,
            lr=learning_rate,
        )
        active_stage = "training"
        publish_training_stage(
            run_registry,
            run_id,
            "lifecycle",
            "training",
            "active",
            "Training step loop active.",
            total_steps=steps,
            progress=0.08,
        )
        for step in range(1, steps + 1):
            if training_task_controller.is_cancel_requested(run_id):
                raise RuntimeError("__cancelled__")
            data_started = _start_stage(
                run_registry,
                run_id,
                "step",
                "data",
                f"Preparing batch for step {step}.",
                step=step,
                steps=steps,
                progress=step / steps,
            )
            images, labels = resource.train_batch(step, batch_size)
            _finish_stage(
                run_registry,
                run_id,
                "step",
                "data",
                f"Batch ready for step {step}.",
                data_started,
                step=step,
                steps=steps,
                progress=step / steps,
            )
            step_start = time.perf_counter()

            def publish_step_stage(stage: str, state: str, duration_ms: float | None) -> None:
                publish_training_stage(
                    run_registry,
                    run_id,
                    "step",
                    stage,
                    state,
                    f"{stage.replace('_', ' ').title()} at step {step}.",
                    step=step,
                    total_steps=steps,
                    duration_ms=duration_ms,
                    progress=step / steps,
                )

            step_result = resource.training_step(model, images, labels, optimizer, step, publish_step_stage)
            step_time_ms = (time.perf_counter() - step_start) * 1000
            last_step = step
            elapsed_sec = time.perf_counter() - start
            should_record = step == 1 or step % telemetry_stride == 0 or step == steps
            layers = None
            telemetry_metrics = None
            if should_record:
                evaluation_started = _start_stage(
                    run_registry,
                    run_id,
                    "milestone",
                    "evaluation",
                    f"Evaluating telemetry probe at step {step}.",
                    step=step,
                    steps=steps,
                    progress=step / steps,
                )
                model.eval()
                telemetry_images = probe_images if resource.task == "detection" else images
                telemetry_targets = probe_targets if resource.task == "detection" else labels
                telemetry = runtime.telemetry_snapshot(model, telemetry_images, telemetry_targets)
                resource_metrics = resource.evaluation_metrics(model, telemetry_images, telemetry_targets, step)
                model.train()
                layers = telemetry.layers
                telemetry_metrics = {**telemetry.metrics, **resource_metrics}
                _finish_stage(
                    run_registry,
                    run_id,
                    "milestone",
                    "evaluation",
                    f"Evaluation recorded at step {step}.",
                    evaluation_started,
                    step=step,
                    steps=steps,
                    progress=step / steps,
                )
            publish_training_progress(
                run_registry,
                run_store,
                run_id,
                step,
                steps,
                learning_rate,
                step_result,
                step_time_ms,
                elapsed_sec,
                layers,
                telemetry_metrics,
                batch_size,
                max_live_layer_samples=max_live_layer_samples,
            )
            save_run_config(run_store, run_id, last_step=step)

        if training_task_controller.is_cancel_requested(run_id):
            raise RuntimeError("__cancelled__")

        save_run_config(run_store, run_id, training_status="checkpointing", last_step=last_step)
        publish_training_stage(
            run_registry,
            run_id,
            "lifecycle",
            "training",
            "completed",
            "Training step loop completed.",
            step=last_step,
            total_steps=steps,
            duration_ms=(time.perf_counter() - start) * 1000,
            progress=1.0,
        )
        publish_run_status(
            run_registry,
            run_id,
            "checkpointing",
            "Saving final checkpoint and probe samples.",
            step=last_step,
            total_steps=steps,
            elapsed_sec=time.perf_counter() - start,
            progress=1.0,
        )
        active_stage = "checkpointing"
        checkpointing_started = _start_stage(
            run_registry,
            run_id,
            "lifecycle",
            "checkpointing",
            "Finalizing run artifacts.",
            step=last_step,
            steps=steps,
            progress=1.0,
        )
        checkpoint_started = _start_stage(
            run_registry,
            run_id,
            "milestone",
            "checkpoint",
            "Saving final checkpoint and probe samples.",
            step=last_step,
            steps=steps,
            progress=1.0,
        )
        sample = io.BytesIO()
        target_key = "labels" if resource.task == "classification" else "targets"
        recorded_predictions: list[dict] = []
        model.eval()
        try:
            with torch.no_grad():
                for index in range(int(probe_images.shape[0])):
                    if isinstance(probe_targets, torch.Tensor):
                        target = int(probe_targets[index].item())
                    else:
                        target = probe_targets[index]
                    result = runtime.inference(
                        model,
                        probe_images[index : index + 1],
                        target,
                        resource.class_names,
                    )
                    recorded_predictions.append(result.output)
        except Exception:
            recorded_predictions = []
        torch.save(
            {
                "images": probe_images,
                target_key: probe_targets,
                "predictions": recorded_predictions,
                "sample_source": sample_source,
                "task": resource.task,
            },
            sample,
        )
        run_store.save_samples(run_id, sample.getvalue())

        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint_path = run_store.save_checkpoint_bytes(run_id, steps, checkpoint.getvalue(), epoch=1)
        fingerprint = fingerprint_state_dict(model.state_dict())
        run_store.record_fingerprint(run_id, checkpoint_path.name, fingerprint)
        _finish_stage(
            run_registry,
            run_id,
            "milestone",
            "checkpoint",
            "Final checkpoint recorded.",
            checkpoint_started,
            step=last_step,
            steps=steps,
            progress=1.0,
        )
        _finish_stage(
            run_registry,
            run_id,
            "lifecycle",
            "checkpointing",
            "Run artifacts finalized.",
            checkpointing_started,
            step=last_step,
            steps=steps,
            progress=1.0,
        )

        entry_meta = run_store.load_entry_meta(run_id)
        save_run_config(
            run_store,
            run_id,
            source="training-resource",
            run_kind="resource-training",
            resource_name=resource.name,
            task=resource.task,
            inference_only=False,
            training_status="completed",
            training_recipe="resource-contract",
            runtime_hooks=resource.runtime_hooks,
            optimizer=optimizer.__class__.__name__,
            steps=steps,
            telemetry_stride=telemetry_stride,
            batch_size=batch_size,
            lr=learning_rate,
            entry_file=entry_meta["entry_file"] if entry_meta else None,
            entry_class=run_store.load_entry_class(run_id),
            input_shape=resource.input_shape,
            classes=resource.classes,
            class_names=resource.class_names,
            dataset_spec=resource.dataset_spec,
            output_schema=resource.output_schema,
            metric_schema=resource.metric_schema,
            data_source=resource.metadata.get("data_source"),
            sample_source=resource.sample_source,
            weights="trained",
            completed_at=time.time(),
            last_step=last_step,
        )
        run_registry.publish(
            run_id,
            [
                build_run_event(
                    run_id,
                    "checkpoint",
                    "checkpoint",
                    last_step,
                    {
                        "path": str(checkpoint_path),
                        "size_mb": round(checkpoint_path.stat().st_size / (1024**2), 3),
                        "fingerprint": fingerprint,
                    },
                ),
            ],
        )
        publish_run_status(
            run_registry,
            run_id,
            "completed",
            "Training completed.",
            step=last_step,
            total_steps=steps,
            elapsed_sec=time.perf_counter() - start,
            progress=1.0,
        )
        publish_training_stage(
            run_registry,
            run_id,
            "lifecycle",
            "completed",
            "completed",
            "Run completed.",
            step=last_step,
            total_steps=steps,
            duration_ms=(time.perf_counter() - start) * 1000,
            progress=1.0,
        )
        run_registry.publish(
            run_id,
            [build_run_event(run_id, "run_complete", "training", last_step, {"status": "trained", "run_kind": "resource-training", "inference_only": False})],
        )
        training_task_controller.finish(run_id, "completed")
    except Exception as exc:
        cancelled = str(exc) == "__cancelled__"
        status = "cancelled" if cancelled else "failed"
        message = "Training cancelled." if cancelled else f"Training failed: {exc}"
        publish_training_stage(
            run_registry,
            run_id,
            "lifecycle",
            active_stage,
            "cancelled" if cancelled else "failed",
            message,
            step=last_step,
            total_steps=steps,
            duration_ms=(time.perf_counter() - start) * 1000,
            progress=(last_step / steps) if steps else 0.0,
        )
        save_run_config(
            run_store,
            run_id,
            training_status=status,
            cancel_requested=cancelled or training_task_controller.is_cancel_requested(run_id),
            error=None if cancelled else str(exc),
            completed_at=time.time(),
            last_step=last_step,
        )
        publish_run_status(
            run_registry,
            run_id,
            status,
            message,
            step=last_step,
            total_steps=steps,
            elapsed_sec=time.perf_counter() - start,
            progress=(last_step / steps) if steps else 0.0,
            extra={"error": None if cancelled else str(exc)},
        )
        run_registry.publish(
            run_id,
            [
                build_run_event(
                    run_id,
                    "run_complete",
                    "training",
                    last_step,
                    {
                        "status": status,
                        "run_kind": "resource-training",
                        "error": None if cancelled else str(exc),
                        "inference_only": False,
                    },
                )
            ],
        )
        training_task_controller.finish(run_id, status)
