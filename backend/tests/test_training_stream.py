import asyncio

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
