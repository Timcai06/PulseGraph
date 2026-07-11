"""Zero-dependency PulseGraph telemetry client for training scripts.

Usage:

    from pulsegraph_client import PulseGraphRun

    with PulseGraphRun(run_id="mnist-mlp") as run:
        run.metric(step=1, epoch=1, loss=2.3, accuracy=0.1)
        run.infra(step=1, device="mps", step_time_ms=34.0)
        run.complete(step=100)

Events are batched and posted from a background thread, so the training loop
never blocks on the network. If the PulseGraph backend is unreachable the
client disables itself after printing a single notice.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8010"
SCHEMA_VERSION = "pulsegraph.event.v1"


class PulseGraphRun:
    def __init__(
        self,
        run_id: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        flush_interval: float = 0.5,
        connect_timeout: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.run_id = run_id or f"run-{time.strftime('%Y%m%d-%H%M%S')}"
        self.flush_interval = flush_interval
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._enabled = self._check_backend(connect_timeout)
        self._worker: threading.Thread | None = None
        if self._enabled:
            self._worker = threading.Thread(target=self._drain_loop, daemon=True)
            self._worker.start()
            print(f"PulseGraph: streaming run '{self.run_id}' to {self.base_url}")
        else:
            print("PulseGraph: backend not reachable, telemetry disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _check_backend(self, timeout: float) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=timeout) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _emit(
        self,
        source: str,
        event_type: str,
        step: int,
        epoch: int | None = None,
        layer: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self._enabled:
            return
        self._queue.put(
            {
                "event_id": str(uuid.uuid4()),
                "schema_version": SCHEMA_VERSION,
                "ts_ns": time.time_ns(),
                "source": source,
                "type": event_type,
                "run_id": self.run_id,
                "step": step,
                "epoch": epoch,
                "layer": layer,
                "payload": payload or {},
            }
        )

    def metric(self, step: int, epoch: int | None = None, **payload: Any) -> None:
        self._emit("training", "metric", step, epoch, payload=payload)

    def layer_snapshot(self, step: int, layer: str, epoch: int | None = None, **payload: Any) -> None:
        self._emit("runtime_hook", "layer_snapshot", step, epoch, layer=layer, payload=payload)

    def infra(self, step: int, epoch: int | None = None, **payload: Any) -> None:
        self._emit("infra", "infra", step, epoch, payload=payload)

    def evidence(self, step: int, kind: str, epoch: int | None = None, **payload: Any) -> None:
        """Record structured investigation evidence without promoting it to a chart metric."""
        self._emit("training", "evidence", step, epoch, payload={"kind": kind, **payload})

    def training_stage(
        self,
        step: int,
        scope: str,
        stage: str,
        state: str,
        message: str,
        epoch: int | None = None,
        **payload: Any,
    ) -> None:
        self._emit(
            "training",
            "training_stage",
            step,
            epoch,
            payload={"scope": scope, "stage": stage, "state": state, "message": message, **payload},
        )

    def checkpoint(self, step: int, epoch: int | None = None, **payload: Any) -> None:
        self._emit("checkpoint", "checkpoint", step, epoch, payload=payload)

    def complete(self, step: int, epoch: int | None = None, status: str = "completed", **payload: Any) -> None:
        self._emit("training", "run_complete", step, epoch, payload={"status": status, **payload})

    def graph(self, tensors: list[dict[str, Any]], step: int = 0, epoch: int | None = None) -> None:
        """Describe the model as {name, shape} tensor specs; the backend infers the graph."""
        self._emit("training", "graph", step, epoch, payload={"tensors": tensors})

    # ---- provenance registration (training-time recording) ----

    def register_source(self, source_code: str, entry_class: str) -> None:
        """Record the model's source so the backend can rebuild it for forward replay."""
        self._emit(
            "training",
            "source_registered",
            step=0,
            payload={"classes": [{"name": entry_class, "source_code": source_code}], "entry_class": entry_class},
        )

    def register_config(self, config: dict[str, Any]) -> None:
        """Record training hyperparameters (lr, epochs, batch size, optimizer, ...)."""
        self._emit("training", "config_registered", step=0, payload={"config": config})

    def register_graph(self, graph_json: dict[str, Any]) -> None:
        """Record the exact fx-traced compute graph (see pulsegraph_torch.trace_model_graph)."""
        self._emit("training", "graph_registered", step=0, payload={"graph": graph_json})
        # also mirror it into the live stream so an attached dashboard updates immediately
        self._emit("training", "graph", step=0, payload={"graph": graph_json})

    # ---- binary artifact uploads ----

    def _post_bytes(self, path: str, data: bytes) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            print(f"PulseGraph: upload to {path} failed")
            return None

    def upload_samples(self, data: bytes) -> bool:
        """Upload serialized probe samples (torch.save of {'images', 'labels'} bytes)."""
        return self._post_bytes(f"/api/runs/{self.run_id}/samples", data) is not None

    def save_checkpoint(self, step: int, epoch: int | None, model: Any, path: str | None = None) -> str | None:
        """Save model.state_dict() as a run checkpoint (uploaded to the backend store).

        Returns the backend-side path, or None when telemetry is disabled/failed.
        Optionally also writes a local copy when `path` is given.
        """
        if not self._enabled:
            return None
        import io

        import torch  # lazy: keep this module importable without torch

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        data = buffer.getvalue()
        if path:
            with open(path, "wb") as handle:
                handle.write(data)
        query = f"step={step}" + (f"&epoch={epoch}" if epoch is not None else "")
        result = self._post_bytes(f"/api/runs/{self.run_id}/checkpoints?{query}", data)
        if result is None:
            return None
        self.checkpoint(
            step,
            epoch,
            path=result.get("path"),
            size_mb=round(len(data) / (1024**2), 3),
            fingerprint=result.get("fingerprint"),
        )
        return result.get("path")

    def training_step(
        self,
        step: int,
        epoch: int | None = None,
        metric: dict[str, Any] | None = None,
        layer: str | None = None,
        layer_snapshot: dict[str, Any] | None = None,
        infra: dict[str, Any] | None = None,
    ) -> None:
        """Emit the common metric/layer/infra bundle for one observed train step."""
        if metric:
            self.metric(step, epoch, **metric)
        if layer and layer_snapshot:
            self.layer_snapshot(step, layer, epoch, **layer_snapshot)
        if infra:
            self.infra(step, epoch, **infra)

    def _post_batch(self, batch: list[dict[str, Any]]) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/api/runs/{self.run_id}/events",
            data=json.dumps(batch).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=2.0).close()
        except (urllib.error.URLError, OSError, ValueError):
            self._enabled = False
            print("PulseGraph: lost connection to backend, telemetry disabled")

    def _drain_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            batch = [item]
            deadline = time.monotonic() + self.flush_interval
            while time.monotonic() < deadline:
                try:
                    extra = self._queue.get(timeout=deadline - time.monotonic())
                except queue.Empty:
                    break
                if extra is None:
                    if batch and self._enabled:
                        self._post_batch(batch)
                    return
                batch.append(extra)
            if self._enabled:
                self._post_batch(batch)

    def close(self, timeout: float = 3.0) -> None:
        if self._worker is None:
            return
        self._queue.put(None)
        self._worker.join(timeout=timeout)
        self._worker = None

    def __enter__(self) -> "PulseGraphRun":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
