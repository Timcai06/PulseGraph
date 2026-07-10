from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.schemas import CheckpointInfo, RunEvent

SAFE_RUN_ID = re.compile(r"[^A-Za-z0-9._-]")
CHECKPOINT_NAME = re.compile(r"step_(\d+)(?:_epoch_(\d+))?\.pt$")


def runs_dir() -> Path:
    override = os.environ.get("PULSEGRAPH_RUNS_DIR")
    base = Path(override) if override else Path(__file__).resolve().parents[2] / "runs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def safe_run_id(run_id: str) -> str:
    return SAFE_RUN_ID.sub("_", run_id) or "run"


class RunStore:
    """Per-run provenance storage: events.jsonl, source, config, graph, samples, checkpoints."""

    def run_dir(self, run_id: str) -> Path:
        directory = runs_dir() / safe_run_id(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def existing_run_dir(self, run_id: str) -> Path:
        return runs_dir() / safe_run_id(run_id)

    def delete_run(self, run_id: str) -> bool:
        deleted = False
        directory = self.existing_run_dir(run_id)
        if directory.exists():
            shutil.rmtree(directory)
            deleted = True
        legacy = runs_dir() / f"{safe_run_id(run_id)}.jsonl"
        if legacy.exists():
            legacy.unlink()
            deleted = True
        return deleted

    def _events_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "events.jsonl"

    # ---- events ----

    def append(self, run_id: str, events: list[RunEvent]) -> None:
        with self._events_path(run_id).open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event.model_dump(), ensure_ascii=False) + "\n")

    def _parse_events(self, lines: list[str]) -> list[RunEvent]:
        events: list[RunEvent] = []
        for line in lines:
            try:
                events.append(RunEvent.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError):
                continue
        return events

    def load_events(self, run_id: str, max_events: int | None = None) -> list[RunEvent]:
        path = self._events_path(run_id)
        if not path.exists():
            # legacy flat layout: runs/{id}.jsonl
            path = runs_dir() / f"{safe_run_id(run_id)}.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        if max_events is not None:
            lines = lines[-max_events:]
        return self._parse_events(lines)

    def load_all(self, max_events_per_run: int) -> dict[str, list[RunEvent]]:
        """Load every persisted run, keeping only the newest events per run.

        Reads both the current layout (runs/{id}/events.jsonl) and the legacy
        flat layout (runs/{id}.jsonl).
        """
        runs: dict[str, list[RunEvent]] = {}
        for path in sorted(runs_dir().glob("*.jsonl")) + sorted(runs_dir().glob("*/events.jsonl")):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            events = self._parse_events(lines[-max_events_per_run:])
            if events:
                runs[events[0].run_id] = events
        return runs

    # ---- provenance files ----

    SAFE_SOURCE_PATH = re.compile(r"^[A-Za-z0-9._/-]+$")
    ALLOWED_SOURCE_SUFFIXES = {".py", ".json", ".png", ".jpg", ".jpeg"}

    def source_dir(self, run_id: str) -> Path:
        directory = self.run_dir(run_id) / "source"
        directory.mkdir(exist_ok=True)
        return directory

    def _write_entry_meta(self, run_id: str, entry_class: str, entry_file: str, origin: str) -> None:
        (self.source_dir(run_id) / "entry.json").write_text(
            json.dumps({"entry_class": entry_class, "entry_file": entry_file, "origin": origin}), encoding="utf-8"
        )

    def save_source(self, run_id: str, source_code: str, entry_class: str | None = None) -> None:
        """Single-file source recorded at training time (register_source event)."""
        (self.source_dir(run_id) / "model.py").write_text(source_code, encoding="utf-8")
        if entry_class:
            self._write_entry_meta(run_id, entry_class, "model.py", "recorded")

    def save_source_files(
        self, run_id: str, files: list[tuple[str, bytes]], entry_file: str, entry_class: str
    ) -> list[str]:
        """User-attached multi-file source/assets; relative paths are preserved under source/."""
        saved: list[str] = []
        source_dir = self.source_dir(run_id)
        for relative_path, content in files:
            normalized = relative_path.replace("\\", "/").lstrip("/")
            suffix = Path(normalized).suffix.lower()
            if (
                ".." in normalized.split("/")
                or not self.SAFE_SOURCE_PATH.match(normalized)
                or suffix not in self.ALLOWED_SOURCE_SUFFIXES
            ):
                continue
            target = source_dir / normalized
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            saved.append(normalized)
        if entry_file in saved:
            self._write_entry_meta(run_id, entry_class, entry_file, "user-attached")
        return saved

    def list_source_files(self, run_id: str) -> list[str]:
        source_dir = self.run_dir(run_id) / "source"
        if not source_dir.exists():
            return []
        return sorted(str(path.relative_to(source_dir)) for path in source_dir.rglob("*.py"))

    def load_entry_meta(self, run_id: str) -> dict[str, str] | None:
        path = self.run_dir(run_id) / "source" / "entry.json"
        if not path.exists():
            return None
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        meta.setdefault("entry_file", "model.py")
        meta.setdefault("origin", "recorded")
        return meta if meta.get("entry_class") else None

    def load_source(self, run_id: str) -> str | None:
        """Entry-file source text (used for display and quick checks)."""
        meta = self.load_entry_meta(run_id)
        entry_file = meta["entry_file"] if meta else "model.py"
        path = self.run_dir(run_id) / "source" / entry_file
        return path.read_text(encoding="utf-8") if path.exists() else None

    def load_source_file(self, run_id: str, relative_path: str) -> str | None:
        source_dir = self.run_dir(run_id) / "source"
        target = (source_dir / relative_path).resolve()
        if not str(target).startswith(str(source_dir.resolve())) or not target.exists():
            return None
        return target.read_text(encoding="utf-8")

    def source_path(self, run_id: str) -> Path | None:
        meta = self.load_entry_meta(run_id)
        entry_file = meta["entry_file"] if meta else "model.py"
        path = self.run_dir(run_id) / "source" / entry_file
        return path if path.exists() else None

    def load_entry_class(self, run_id: str) -> str | None:
        meta = self.load_entry_meta(run_id)
        return meta["entry_class"] if meta else None

    def save_config(self, run_id: str, config: dict[str, Any]) -> None:
        (self.run_dir(run_id) / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load_config(self, run_id: str) -> dict[str, Any] | None:
        path = self.run_dir(run_id) / "config.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def save_graph(self, run_id: str, graph_json: dict[str, Any]) -> None:
        (self.run_dir(run_id) / "graph.json").write_text(json.dumps(graph_json, ensure_ascii=False), encoding="utf-8")

    def load_graph(self, run_id: str) -> dict[str, Any] | None:
        path = self.run_dir(run_id) / "graph.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    # ---- probe samples ----

    def save_samples(self, run_id: str, data: bytes) -> Path:
        path = self.run_dir(run_id) / "samples.pt"
        path.write_bytes(data)
        return path

    def samples_path(self, run_id: str) -> Path | None:
        path = self.run_dir(run_id) / "samples.pt"
        return path if path.exists() else None

    # ---- checkpoints ----

    def checkpoints_dir(self, run_id: str) -> Path:
        directory = self.run_dir(run_id) / "checkpoints"
        directory.mkdir(exist_ok=True)
        return directory

    def save_checkpoint_bytes(self, run_id: str, step: int, data: bytes, epoch: int | None = None) -> Path:
        suffix = f"step_{step:04d}" + (f"_epoch_{epoch}" if epoch is not None else "")
        path = self.checkpoints_dir(run_id) / f"{suffix}.pt"
        path.write_bytes(data)
        return path

    def list_checkpoints(self, run_id: str) -> list[CheckpointInfo]:
        fingerprints = self._load_fingerprints(run_id)
        checkpoints: list[CheckpointInfo] = []
        for path in sorted(self.checkpoints_dir(run_id).glob("*.pt")):
            match = CHECKPOINT_NAME.search(path.name)
            if not match:
                continue
            checkpoints.append(
                CheckpointInfo(
                    step=int(match.group(1)),
                    epoch=int(match.group(2)) if match.group(2) else None,
                    path=str(path.relative_to(runs_dir())),
                    size_mb=round(path.stat().st_size / (1024**2), 3),
                    fingerprint=fingerprints.get(path.name),
                )
            )
        checkpoints.sort(key=lambda info: info.step)
        return checkpoints

    def checkpoint_path(self, run_id: str, step: int | None = None) -> Path | None:
        """Path for the checkpoint at `step`, or the latest when step is None/0."""
        checkpoints = self.list_checkpoints(run_id)
        if not checkpoints:
            return None
        if not step:
            chosen = checkpoints[-1]
        else:
            chosen = next((info for info in checkpoints if info.step == step), None)
            if chosen is None:
                return None
        return runs_dir() / chosen.path

    # ---- weights fingerprint index ----

    def _fingerprints_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "fingerprints.json"

    def _load_fingerprints(self, run_id: str) -> dict[str, str]:
        path = self._fingerprints_path(run_id)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def record_fingerprint(self, run_id: str, checkpoint_name: str, fingerprint: str) -> None:
        fingerprints = self._load_fingerprints(run_id)
        fingerprints[checkpoint_name] = fingerprint
        self._fingerprints_path(run_id).write_text(json.dumps(fingerprints, indent=2), encoding="utf-8")

    def find_run_by_fingerprint(self, fingerprint: str) -> tuple[str, int] | None:
        """Return (run_id, checkpoint_step) whose recorded fingerprint matches."""
        for run_path in runs_dir().iterdir():
            if not run_path.is_dir():
                continue
            run_id = run_path.name
            for name, recorded in self._load_fingerprints(run_id).items():
                if recorded == fingerprint:
                    match = CHECKPOINT_NAME.search(name)
                    return run_id, int(match.group(1)) if match else 0
        return None
