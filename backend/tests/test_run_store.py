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


def test_load_first_event_reads_head_of_log_past_bounded_window(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PULSEGRAPH_RUNS_DIR", str(tmp_path))
    store = RunStore()
    run_id = "graph-head-run"
    graph_event = RunEvent(
        event_id=str(uuid.uuid4()),
        ts_ns=time.time_ns(),
        source="training",
        type="graph",
        run_id=run_id,
        step=0,
        payload={"graph": {"nodes": [{"id": "backbone.0"}], "edges": []}},
    )
    store.append(run_id, [graph_event])
    store.append(run_id, [_event(run_id, step) for step in range(1, 21)])

    # bounded tail read drops the head-of-log graph event...
    tail = store.load_events(run_id, max_events=5)
    assert all(event.type != "graph" for event in tail)

    # ...but the head scan still finds it
    found = store.load_first_event(run_id, "graph")
    assert found is not None
    assert found.event_id == graph_event.event_id


def test_load_first_event_returns_none_when_type_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PULSEGRAPH_RUNS_DIR", str(tmp_path))
    store = RunStore()
    run_id = "no-graph-run"
    store.append(run_id, [_event(run_id, 1)])

    assert store.load_first_event(run_id, "graph") is None
    assert store.load_first_event("missing-run", "graph") is None
