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

    def checkpoint(self, step: int, epoch: int | None = None, **payload: Any) -> None:
        self._emit("checkpoint", "checkpoint", step, epoch, payload=payload)

    def complete(self, step: int, epoch: int | None = None) -> None:
        self._emit("training", "run_complete", step, epoch)

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
