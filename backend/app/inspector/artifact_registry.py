from __future__ import annotations

import hashlib
from pathlib import Path

from app.schemas import ArtifactRecord


def register_artifact(path: Path, trust_level: str = "untrusted") -> ArtifactRecord:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    sha256 = digest.hexdigest()
    suffix = path.suffix.lower().lstrip(".") or "unknown"
    return ArtifactRecord(
        artifact_id=f"pulsegraph:sha256:{sha256[:16]}",
        path=str(path),
        filename=path.name,
        sha256=sha256,
        size_bytes=path.stat().st_size,
        format=suffix,
        trust_level=trust_level,  # type: ignore[arg-type]
    )
