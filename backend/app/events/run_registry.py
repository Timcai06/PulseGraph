from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.schemas import RunEvent, RunSummary

BUFFER_SIZE = 1024
KEEPALIVE_SECONDS = 15.0
MAX_RUNS = 64
SUBSCRIBER_QUEUE_SIZE = 256


@dataclass
class LiveRun:
    run_id: str
    created_at: float = field(default_factory=time.time)
    last_event_at: float = field(default_factory=time.time)
    completed: bool = False
    event_count: int = 0
    last_step: int = 0
    buffer: deque[RunEvent] = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    subscribers: set[asyncio.Queue[RunEvent]] = field(default_factory=set)


class RunRegistry:
    """In-memory registry of live training runs fed by the ingest endpoint."""

    def __init__(self, max_runs: int = MAX_RUNS, subscriber_queue_size: int = SUBSCRIBER_QUEUE_SIZE) -> None:
        self._runs: dict[str, LiveRun] = {}
        self.max_runs = max_runs
        self.subscriber_queue_size = subscriber_queue_size

    def _ensure_capacity(self, run_id: str) -> None:
        if run_id in self._runs or len(self._runs) < self.max_runs:
            return
        oldest = min(self._runs.values(), key=lambda run: run.created_at)
        del self._runs[oldest.run_id]

    def publish(self, run_id: str, events: list[RunEvent]) -> LiveRun:
        self._ensure_capacity(run_id)
        run = self._runs.setdefault(run_id, LiveRun(run_id=run_id))
        for event in events:
            run.buffer.append(event)
            run.event_count += 1
            run.last_step = max(run.last_step, event.step)
            run.last_event_at = time.time()
            if event.type == "run_complete":
                run.completed = True
            for queue in run.subscribers:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    continue
        return run

    def get(self, run_id: str) -> LiveRun | None:
        return self._runs.get(run_id)

    def list_runs(self) -> list[RunSummary]:
        runs = sorted(self._runs.values(), key=lambda run: run.created_at, reverse=True)
        return [
            RunSummary(
                run_id=run.run_id,
                created_at=run.created_at,
                last_event_at=run.last_event_at,
                completed=run.completed,
                event_count=run.event_count,
                last_step=run.last_step,
            )
            for run in runs
        ]

    async def subscribe(self, run_id: str) -> AsyncIterator[RunEvent | None]:
        """Replay buffered events, then yield live ones; None marks a keepalive tick."""
        run = self._runs.setdefault(run_id, LiveRun(run_id=run_id))
        queue: asyncio.Queue[RunEvent] = asyncio.Queue(maxsize=self.subscriber_queue_size)
        run.subscribers.add(queue)
        try:
            for event in list(run.buffer):
                yield event
            if run.completed:
                return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    yield None
                    continue
                yield event
                if event.type == "run_complete":
                    return
        finally:
            run.subscribers.discard(queue)


run_registry = RunRegistry()
