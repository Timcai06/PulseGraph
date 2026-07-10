import threading
import time

from app.events.training_control import TrainingTaskController, training_task_controller


def test_training_controller_register_is_idempotent_for_same_run() -> None:
    controller = TrainingTaskController(max_concurrent_runs=1)

    first = controller.register("run-a")
    second = controller.register("run-a")

    assert first.queue_position == 1
    assert second.queue_position == 1


def test_training_controller_second_register_does_not_reorder_waiting_run() -> None:
    controller = TrainingTaskController(max_concurrent_runs=1)
    controller.register("run-a")
    controller.register("run-b")
    acquired = controller.acquire("run-a")
    assert acquired.active is True

    before = controller.get("run-b")
    assert before is not None
    assert before.queue_position == 1

    controller.register("run-b")
    after = controller.get("run-b")

    assert after is not None
    assert after.queue_position == 1
    controller.finish("run-a", "completed")


def test_global_training_controller_keeps_default_local_concurrency_to_one() -> None:
    assert training_task_controller.max_concurrent_runs == 1


def test_training_controller_serializes_waiting_run_until_slot_frees() -> None:
    controller = TrainingTaskController(max_concurrent_runs=1)
    controller.register("run-a")
    controller.register("run-b")
    controller.acquire("run-a")
    acquired_second = threading.Event()

    def wait_for_second() -> None:
        controller.acquire("run-b")
        acquired_second.set()

    thread = threading.Thread(target=wait_for_second, daemon=True)
    thread.start()
    time.sleep(0.05)
    assert acquired_second.is_set() is False

    controller.finish("run-a", "completed")
    thread.join(timeout=1)

    assert acquired_second.is_set() is True
