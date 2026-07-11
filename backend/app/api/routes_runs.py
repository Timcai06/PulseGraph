from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.events.run_registry import RunRegistry
from app.events.run_store import RunStore
from app.reports.comparison import compare_runs
from app.schemas import RunComparison, RunSummary


def create_runs_router(run_registry: RunRegistry, run_store: RunStore) -> APIRouter:
    router = APIRouter(prefix="/api/runs", tags=["runs"])

    @router.get("")
    def list_runs() -> list[RunSummary]:
        return run_registry.list_runs()

    @router.get("/compare")
    def compare_recorded_runs(baseline_run_id: str, candidate_run_id: str) -> RunComparison:
        comparison = compare_runs(run_store, baseline_run_id, candidate_run_id)
        if comparison is None:
            raise HTTPException(status_code=404, detail="Both recorded runs are required for comparison.")
        return comparison

    @router.delete("/{run_id}")
    def delete_run(run_id: str):
        memory_deleted = run_registry.delete(run_id)
        disk_deleted = run_store.delete_run(run_id)
        if not memory_deleted and not disk_deleted:
            raise HTTPException(status_code=404, detail="Run not found.")
        return {"run_id": run_id, "deleted": True}

    return router
