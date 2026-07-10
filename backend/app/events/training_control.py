from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


DEFAULT_MAX_CONCURRENT_TRAINING_RUNS = 1


@dataclass
class TrainingTaskState:
    run_id: str
    phase: str = "queued"
    created_at: float = field(default_factory=time.time)
    acquired_at: float | None = None
    finished_at: float | None = None
    queue_position: int = 0
    active: bool = False
    completed: bool = False
    cancel_requested: bool = False


class TrainingTaskController:
    """Small in-process queue for local training jobs with cooperative cancel."""

    def __init__(self, max_concurrent_runs: int = DEFAULT_MAX_CONCURRENT_TRAINING_RUNS) -> None:
        self.max_concurrent_runs = max(1, int(max_concurrent_runs))
        self._condition = threading.Condition()
        self._active_runs: set[str] = set()
        self._queue: list[str] = []
        self._tasks: dict[str, TrainingTaskState] = {}

    def register(self, run_id: str) -> TrainingTaskState:
        with self._condition:
            state = self._tasks.get(run_id)
            if state is None:
                state = TrainingTaskState(run_id=run_id)
                self._tasks[run_id] = state
            if run_id not in self._queue and run_id not in self._active_runs and not state.completed:
                self._queue.append(run_id)
            self._refresh_queue_positions()
            return self._clone_state(state)

    def acquire(self, run_id: str, poll_interval_sec: float = 0.1) -> TrainingTaskState:
        with self._condition:
            if run_id not in self._tasks:
                self.register(run_id)
            while True:
                state = self._tasks[run_id]
                if state.cancel_requested:
                    state.phase = "cancelled"
                    state.finished_at = state.finished_at or time.time()
                    state.completed = True
                    self._remove_from_queue(run_id)
                    self._condition.notify_all()
                    return self._clone_state(state)
                can_start = (
                    len(self._active_runs) < self.max_concurrent_runs
                    and self._queue
                    and self._queue[0] == run_id
                )
                if can_start:
                    self._queue.pop(0)
                    self._active_runs.add(run_id)
                    state.active = True
                    state.phase = "running"
                    state.acquired_at = time.time()
                    state.queue_position = 0
                    self._refresh_queue_positions()
                    return self._clone_state(state)
                self._refresh_queue_positions()
                self._condition.wait(timeout=poll_interval_sec)

    def request_cancel(self, run_id: str) -> TrainingTaskState | None:
        with self._condition:
            state = self._tasks.get(run_id)
            if state is None:
                return None
            state.cancel_requested = True
            if not state.completed:
                state.phase = "cancelling" if state.active else "cancel_requested"
            self._condition.notify_all()
            return self._clone_state(state)

    def finish(self, run_id: str, phase: str) -> TrainingTaskState | None:
        with self._condition:
            state = self._tasks.get(run_id)
            if state is None:
                return None
            self._active_runs.discard(run_id)
            self._remove_from_queue(run_id)
            state.active = False
            state.completed = True
            state.phase = phase
            state.finished_at = time.time()
            self._refresh_queue_positions()
            self._condition.notify_all()
            return self._clone_state(state)

    def get(self, run_id: str) -> TrainingTaskState | None:
        with self._condition:
            state = self._tasks.get(run_id)
            return self._clone_state(state) if state is not None else None

    def is_cancel_requested(self, run_id: str) -> bool:
        with self._condition:
            state = self._tasks.get(run_id)
            return bool(state and state.cancel_requested)

    def _remove_from_queue(self, run_id: str) -> None:
        self._queue = [queued for queued in self._queue if queued != run_id]

    def _refresh_queue_positions(self) -> None:
        for state in self._tasks.values():
            state.queue_position = 0
        for index, queued_run_id in enumerate(self._queue, start=1):
            state = self._tasks.get(queued_run_id)
            if state is not None:
                state.queue_position = index

    @staticmethod
    def _clone_state(state: TrainingTaskState) -> TrainingTaskState:
        return TrainingTaskState(
            run_id=state.run_id,
            phase=state.phase,
            created_at=state.created_at,
            acquired_at=state.acquired_at,
            finished_at=state.finished_at,
            queue_position=state.queue_position,
            active=state.active,
            completed=state.completed,
            cancel_requested=state.cancel_requested,
        )


training_task_controller = TrainingTaskController()
