import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from app.events.run_store import RunStore
from app.schemas import RunEvent


def _event(run_id: str, step: int) -> RunEvent:
    return RunEvent(
        event_id=str(uuid.uuid4()),
        ts_ns=time.time_ns(),
        source="training",
        type="metric",
        run_id=run_id,
        step=step,
        payload={"loss": 1 / step},
    )


def test_concurrent_event_batches_remain_complete(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PULSEGRAPH_RUNS_DIR", str(tmp_path))
    store = RunStore()
    run_id = "concurrent-run"

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda step: store.append(run_id, [_event(run_id, step)]), range(1, 201)))

    events = store.load_events(run_id)
    assert len(events) == 200
    assert {event.step for event in events} == set(range(1, 201))


def test_bounded_event_read_returns_only_latest_lines(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PULSEGRAPH_RUNS_DIR", str(tmp_path))
    store = RunStore()
    run_id = "bounded-run"
    store.append(run_id, [_event(run_id, step) for step in range(1, 21)])

    events = store.load_events(run_id, max_events=5)

    assert [event.step for event in events] == [16, 17, 18, 19, 20]


def test_unknown_event_read_does_not_create_a_run_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PULSEGRAPH_RUNS_DIR", str(tmp_path))
    store = RunStore()

    assert store.load_events("missing", max_events=1) == []
    assert not (tmp_path / "missing").exists()


def test_config_updates_do_not_lose_concurrent_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PULSEGRAPH_RUNS_DIR", str(tmp_path))
    store = RunStore()
    run_id = "config-run"

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: store.update_config(run_id, {f"field_{index}": index}), range(40)))

    config = store.load_config(run_id)
    assert config is not None
    assert all(config[f"field_{index}"] == index for index in range(40))
    assert not list((tmp_path / run_id).glob("*.tmp"))
