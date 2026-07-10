from __future__ import annotations

import asyncio
import json
import math
import random
import time
import uuid
from collections.abc import AsyncIterator

from app.schemas import RunEvent


async def demo_training_events(run_id: str = "demo-run", steps: int = 80) -> AsyncIterator[RunEvent]:
    random.seed(7)
    start = time.perf_counter()
    for step in range(1, steps + 1):
        epoch = 1 + (step - 1) // 25
        loss = max(0.05, 2.25 * math.exp(-step / 24) + random.random() * 0.035)
        accuracy = min(0.985, 0.12 + (1 - math.exp(-step / 23)) * 0.86 + random.random() * 0.01)
        step_time_ms = 34 + math.sin(step / 5) * 5 + random.random() * 2
        samples_per_sec = 128 / (step_time_ms / 1000)
        elapsed = time.perf_counter() - start

        yield RunEvent(
            event_id=str(uuid.uuid4()),
            ts_ns=time.time_ns(),
            source="training",
            type="run_status",
            run_id=run_id,
            step=step,
            epoch=epoch,
            payload={
                "phase": "training",
                "message": f"Training step {step}/{steps}",
                "step": step,
                "total_steps": steps,
                "elapsed_sec": round(elapsed, 2),
                "eta_sec": round((elapsed / step) * max(steps - step, 0), 2),
                "progress": round(step / steps, 4),
            },
        )
        yield RunEvent(
            event_id=str(uuid.uuid4()),
            ts_ns=time.time_ns(),
            source="training",
            type="metric",
            run_id=run_id,
            step=step,
            epoch=epoch,
            payload={
                "loss": round(loss, 4),
                "accuracy": round(accuracy, 4),
                "learning_rate": 0.001,
            },
        )
        yield RunEvent(
            event_id=str(uuid.uuid4()),
            ts_ns=time.time_ns(),
            source="runtime_hook",
            type="layer_snapshot",
            run_id=run_id,
            step=step,
            epoch=epoch,
            layer="linear1",
            payload={
                "activation_mean": round(0.18 + accuracy * 0.36, 4),
                "activation_sparsity": round(max(0.08, 0.48 - accuracy * 0.24), 4),
                "gradient_norm": round(max(0.04, loss * 1.7), 4),
                "weight_std": round(0.055 + step * 0.00018, 4),
            },
        )
        yield RunEvent(
            event_id=str(uuid.uuid4()),
            ts_ns=time.time_ns(),
            source="infra",
            type="infra",
            run_id=run_id,
            step=step,
            epoch=epoch,
            payload={
                "device": "mps",
                "step_time_ms": round(step_time_ms, 2),
                "samples_per_sec": round(samples_per_sec, 1),
                "memory_peak_mb": round(360 + step * 1.7 + random.random() * 4, 1),
                "elapsed_sec": round(elapsed, 2),
            },
        )
        if step % 20 == 0:
            yield RunEvent(
                event_id=str(uuid.uuid4()),
                ts_ns=time.time_ns(),
                source="checkpoint",
                type="checkpoint",
                run_id=run_id,
                step=step,
                epoch=epoch,
                payload={
                    "path": f"runs/{run_id}/checkpoints/step_{step:04d}.pt",
                    "size_mb": round(0.42 + step * 0.003, 3),
                    "save_time_ms": round(12 + random.random() * 5, 2),
                },
            )
        yield RunEvent(
            event_id=str(uuid.uuid4()),
            ts_ns=time.time_ns(),
            source="animation",
            type="animation",
            run_id=run_id,
            step=step,
            epoch=epoch,
            payload={
                "name": "forward_pass" if step % 2 else "backward_pass",
                "path": ["input", "flatten", "linear1", "relu1", "linear2", "softmax"],
            },
        )
        await asyncio.sleep(0.08)

    yield RunEvent(
        event_id=str(uuid.uuid4()),
        ts_ns=time.time_ns(),
        source="training",
        type="run_status",
        run_id=run_id,
        step=steps,
        epoch=1 + (steps - 1) // 25,
        payload={
            "phase": "completed",
            "message": "Training completed.",
            "step": steps,
            "total_steps": steps,
            "elapsed_sec": round(time.perf_counter() - start, 2),
            "progress": 1.0,
        },
    )

    yield RunEvent(
        event_id=str(uuid.uuid4()),
        ts_ns=time.time_ns(),
        source="training",
        type="run_complete",
        run_id=run_id,
        step=steps,
        epoch=1 + (steps - 1) // 25,
        payload={},
    )


def to_sse(event: RunEvent) -> str:
    return f"event: {event.type}\ndata: {json.dumps(event.model_dump(), ensure_ascii=False)}\n\n"
