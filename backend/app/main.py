from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.events.run_registry import run_registry
from app.events.training_stream import demo_training_events, to_sse
from app.inspector.pt_inspector import inspect_pt_file
from app.runtime.demo_mlp import demo_graph, run_demo_forward
from app.schemas import RunEvent, RunSummary


app = FastAPI(title="PulseGraph API", version="0.1.0")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "product": "PulseGraph"}


@app.get("/api/demo/model")
def get_demo_model():
    return demo_graph()


@app.get("/api/demo/forward")
def get_demo_forward(index: int = 0):
    return run_demo_forward(index)


@app.get("/api/runs/demo/stream")
async def stream_demo_run():
    async def generate():
        async for event in demo_training_events():
            yield to_sse(event)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/runs")
def list_runs() -> list[RunSummary]:
    return run_registry.list_runs()


@app.post("/api/runs/{run_id}/events")
def ingest_run_events(run_id: str, events: RunEvent | list[RunEvent]):
    batch = events if isinstance(events, list) else [events]
    for event in batch:
        event.run_id = run_id
    run = run_registry.publish(run_id, batch)
    return {"accepted": len(batch), "run_id": run_id, "completed": run.completed}


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str):
    async def generate():
        async for event in run_registry.subscribe(run_id):
            if event is None:
                yield ": keepalive\n\n"
            else:
                yield to_sse(event)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/inspect/upload")
async def inspect_upload(file: UploadFile = File(...)):
    filename = file.filename or "model.pt"
    suffix = Path(filename).suffix or ".pt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
        total = 0
        while chunk := await file.read(UPLOAD_CHUNK_BYTES):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Uploaded model file is larger than 100 MB.")
            handle.write(chunk)
        handle.flush()
        return inspect_pt_file(Path(handle.name), display_filename=filename).model_dump()
