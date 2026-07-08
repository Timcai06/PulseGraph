import time
import uuid

from fastapi.testclient import TestClient

from app.main import app
from app.events.run_registry import RunRegistry
from app.schemas import RunEvent


client = TestClient(app)


def _event(run_id: str, event_type: str = "metric", step: int = 1, **payload) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "ts_ns": time.time_ns(),
        "source": "training",
        "type": event_type,
        "run_id": run_id,
        "step": step,
        "epoch": 1,
        "payload": payload,
    }


def _run_event(run_id: str, event_type: str = "metric", step: int = 1, **payload) -> RunEvent:
    return RunEvent.model_validate(_event(run_id, event_type, step, **payload))


def test_ingest_single_event_and_list_runs() -> None:
    run_id = f"test-{uuid.uuid4().hex[:8]}"

    response = client.post(f"/api/runs/{run_id}/events", json=_event(run_id, loss=1.2, accuracy=0.4))

    assert response.status_code == 200
    assert response.json() == {"accepted": 1, "run_id": run_id, "completed": False}
    runs = client.get("/api/runs").json()
    match = next(run for run in runs if run["run_id"] == run_id)
    assert match["event_count"] == 1
    assert match["completed"] is False


def test_ingest_batch_overrides_run_id_and_marks_completion() -> None:
    run_id = f"test-{uuid.uuid4().hex[:8]}"
    batch = [
        _event("mismatched-id", step=1, loss=1.0),
        _event("mismatched-id", event_type="run_complete", step=2),
    ]

    response = client.post(f"/api/runs/{run_id}/events", json=batch)

    assert response.status_code == 200
    assert response.json()["accepted"] == 2
    assert response.json()["completed"] is True
    runs = client.get("/api/runs").json()
    match = next(run for run in runs if run["run_id"] == run_id)
    assert match["last_step"] == 2


def test_stream_replays_ingested_events_until_completion() -> None:
    run_id = f"test-{uuid.uuid4().hex[:8]}"
    client.post(
        f"/api/runs/{run_id}/events",
        json=[
            _event(run_id, step=1, loss=0.9, accuracy=0.5),
            _event(run_id, event_type="infra", step=1, device="cpu"),
            _event(run_id, event_type="run_complete", step=1),
        ],
    )

    with client.stream("GET", f"/api/runs/{run_id}/stream") as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: metric" in body
    assert "event: infra" in body
    assert "event: run_complete" in body
    assert f'"run_id": "{run_id}"' in body


def test_registry_evicts_oldest_runs_when_capacity_is_reached() -> None:
    registry = RunRegistry(max_runs=2)

    registry.publish("run-a", [_run_event("run-a", step=1, loss=1.0)])
    registry.publish("run-b", [_run_event("run-b", step=1, loss=0.9)])
    registry.publish("run-c", [_run_event("run-c", step=1, loss=0.8)])

    assert registry.get("run-a") is None
    assert [run.run_id for run in registry.list_runs()] == ["run-c", "run-b"]


def test_registry_drops_for_slow_subscribers_instead_of_growing_unbounded() -> None:
    registry = RunRegistry(subscriber_queue_size=1)
    run = registry.publish("run-a", [_run_event("run-a", step=1, loss=1.0)])
    subscriber = __import__("asyncio").Queue(maxsize=1)
    run.subscribers.add(subscriber)
    subscriber.put_nowait(_run_event("run-a", step=1, loss=1.0))

    registry.publish("run-a", [_run_event("run-a", step=2, loss=0.9)])

    assert subscriber.qsize() == 1
