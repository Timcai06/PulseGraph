import asyncio
from pathlib import Path
from types import SimpleNamespace

import app.main as main_module
from app.events.training_stream import demo_training_events, to_sse


async def _collect_events(steps: int):
    return [event async for event in demo_training_events(run_id="test-run", steps=steps)]


def test_demo_training_events_include_metrics_infra_and_completion() -> None:
    events = asyncio.run(_collect_events(2))

    assert events[0].type == "metric"
    assert events[0].schema_version == "pulsegraph.event.v1"
    assert any(event.type == "infra" for event in events)
    assert events[-1].type == "run_complete"
    assert events[-1].step == 2


def test_to_sse_uses_named_event_and_json_payload() -> None:
    events = asyncio.run(_collect_events(1))

    message = to_sse(events[0])

    assert message.startswith("event: metric\n")
    assert '"schema_version": "pulsegraph.event.v1"' in message
    assert message.endswith("\n\n")


def test_source_training_events_are_published_in_live_batches(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_publish(run_id: str, events):
        calls.append([event.type for event in events])
        return None

    monkeypatch.setattr(main_module.run_registry, "publish", fake_publish)
    monkeypatch.setattr(main_module, "SOURCE_TRAIN_EVENT_INTERVAL_SEC", 0)
    checkpoint = tmp_path / "step_0002.pt"
    checkpoint.write_bytes(b"checkpoint")
    graph = SimpleNamespace(model_dump=lambda: {"nodes": [], "edges": []})

    main_module._publish_source_training_events(
        "train-test",
        graph,
        [
            {"step": 1, "loss": 1.2, "accuracy": 0.2, "learning_rate": 0.001, "step_time_ms": 1.0, "samples_per_sec": 8, "elapsed_sec": 0.1},
            {"step": 2, "loss": 0.8, "accuracy": 0.4, "learning_rate": 0.001, "step_time_ms": 1.0, "samples_per_sec": 8, "elapsed_sec": 0.2},
        ],
        [(1, []), (2, [])],
        Path(checkpoint),
        "abc123",
    )

    assert calls[0] == ["graph"]
    assert ["metric", "infra"] in calls
    assert calls[-1] == ["checkpoint", "run_complete"]
    assert len(calls) >= 4
