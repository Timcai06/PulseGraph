from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from threading import RLock

from app.events.run_store import RunStore
from app.schemas import RunEvent, RunSummary

BUFFER_SIZE = 1024
KEEPALIVE_SECONDS = 15.0
MAX_RUNS = 64
SUBSCRIBER_QUEUE_SIZE = 256


class RunRegistryFullError(RuntimeError):
    pass


@dataclass
class LiveRun:
    run_id: str
    created_at: float = field(default_factory=time.time)
    last_event_at: float = field(default_factory=time.time)
    completed: bool = False
    event_count: int = 0
    last_step: int = 0
    buffer: deque[RunEvent] = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    # Pinned outside the ring buffer: long runs (> BUFFER_SIZE events) evict the
    # step-0 graph event, and without it subscribers can never map layer
    # snapshots onto graph nodes.
    graph_event: RunEvent | None = None
    subscribers: set[LiveSubscriber] = field(default_factory=set)


@dataclass(frozen=True)
class LiveSubscriber:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[RunEvent]


class RunRegistry:
    """In-memory registry of live training runs fed by the ingest endpoint."""

    def __init__(
        self,
        max_runs: int = MAX_RUNS,
        subscriber_queue_size: int = SUBSCRIBER_QUEUE_SIZE,
        store: RunStore | None = None,
    ) -> None:
        self._runs: dict[str, LiveRun] = {}
        self.max_runs = max_runs
        self.subscriber_queue_size = subscriber_queue_size
        self.store = store
        self._lock = RLock()

    def load_from_store(self) -> int:
        """Restore persisted runs into memory; returns how many were loaded."""
        if self.store is None:
            return 0
        for run_id, events in self.store.load_all(BUFFER_SIZE).items():
            self._ensure_capacity(run_id)
            run = self._from_events(run_id, events)
            self._pin_graph_event(run, run_id)
            self._runs[run_id] = run
        return len(self._runs)

    @staticmethod
    def _from_events(run_id: str, events: list[RunEvent]) -> LiveRun:
        run = LiveRun(run_id=run_id)
        run.created_at = events[0].ts_ns / 1e9
        run.last_event_at = events[-1].ts_ns / 1e9
        run.event_count = len(events)
        run.last_step = max(event.step for event in events)
        run.completed = any(event.type == "run_complete" for event in events)
        run.buffer.extend(events)
        return run

    def _pin_graph_event(self, run: LiveRun, run_id: str) -> None:
        """Make the structural graph event survive ring-buffer eviction."""
        if run.graph_event is not None:
            return
        buffered = next((event for event in run.buffer if event.type == "graph"), None)
        if buffered is not None:
            run.graph_event = buffered
            return
        if self.store is not None:
            run.graph_event = self.store.load_first_event(run_id, "graph")

    def ensure(self, run_id: str) -> LiveRun | None:
        with self._lock:
            run = self._runs.get(run_id)
        if run is not None or self.store is None:
            return run
        events = self.store.load_events(run_id, max_events=BUFFER_SIZE)
        if not events:
            return None
        with self._lock:
            existing = self._runs.get(run_id)
            if existing is not None:
                return existing
            self._ensure_capacity(run_id)
            run = self._from_events(run_id, events)
            self._pin_graph_event(run, run_id)
            self._runs[run_id] = run
            return run

    def _ensure_capacity(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._runs or len(self._runs) < self.max_runs:
                return
            inactive = [run for run in self._runs.values() if not run.subscribers]
            if not inactive:
                raise RunRegistryFullError("All in-memory runs have active subscribers.")
            oldest = min(inactive, key=lambda run: run.created_at)
            del self._runs[oldest.run_id]

    @staticmethod
    def _notify(subscriber: LiveSubscriber, event: RunEvent) -> None:
        if subscriber.loop.is_closed():
            return

        def deliver() -> None:
            try:
                subscriber.queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

        subscriber.loop.call_soon_threadsafe(deliver)

    def publish(self, run_id: str, events: list[RunEvent]) -> LiveRun:
        with self._lock:
            self._ensure_capacity(run_id)
            run = self._runs.setdefault(run_id, LiveRun(run_id=run_id))
            if self.store is not None:
                self.store.append(run_id, events)
            for event in events:
                run.buffer.append(event)
                run.event_count += 1
                run.last_step = max(run.last_step, event.step)
                run.last_event_at = time.time()
                if event.type == "graph":
                    run.graph_event = event
                if event.type == "run_complete":
                    run.completed = True
                for subscriber in tuple(run.subscribers):
                    self._notify(subscriber, event)
            return run

    def get(self, run_id: str) -> LiveRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def delete(self, run_id: str) -> bool:
        with self._lock:
            return self._runs.pop(run_id, None) is not None

    def list_runs(self) -> list[RunSummary]:
        with self._lock:
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
        run = self.ensure(run_id)
        if run is None:
            raise KeyError(run_id)
        queue: asyncio.Queue[RunEvent] = asyncio.Queue(maxsize=self.subscriber_queue_size)
        subscriber = LiveSubscriber(loop=asyncio.get_running_loop(), queue=queue)
        with self._lock:
            run.subscribers.add(subscriber)
            replay = list(run.buffer)
            completed = run.completed
            graph_event = run.graph_event
        try:
            # Ring buffer may have evicted the step-0 graph event on long runs;
            # without it the client cannot map layer snapshots to graph nodes.
            if graph_event is not None and not any(event.type == "graph" for event in replay):
                yield graph_event
            for event in replay:
                yield event
            if completed:
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
            with self._lock:
                run.subscribers.discard(subscriber)


run_registry = RunRegistry(store=RunStore())
