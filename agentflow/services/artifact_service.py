from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from agentflow.config import get_artifact_storage_dir
from agentflow.db.models import AgentRun, RunArtifact, utc_now
from agentflow.db.session import create_session_factory


class ArtifactError(RuntimeError):
    """Base error for artifact operations."""


class ArtifactRunNotFoundError(ArtifactError):
    pass


class ArtifactNotFoundError(ArtifactError):
    pass


class ArtifactFileMissingError(ArtifactError):
    pass


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: uuid.UUID
    run_id: uuid.UUID
    artifact_type: str
    name: str
    file_path: str
    mime_type: str | None
    size_bytes: int | None
    description: str | None
    created_at: datetime


def save_run_artifact(
    run_id: uuid.UUID,
    *,
    name: str,
    artifact_type: str,
    content: str | bytes,
    mime_type: str | None = None,
    description: str | None = None,
    storage_root: Path | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> ArtifactRecord:
    artifact_id = uuid.uuid4()
    safe_name = _safe_filename(name)
    artifact_type = artifact_type.strip() or "artifact"
    storage_root = storage_root or get_artifact_storage_dir()
    target_dir = storage_root / str(run_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{artifact_id}_{safe_name}"
    data = content if isinstance(content, bytes) else content.encode("utf-8")
    target_path.write_bytes(data)
    size_bytes = target_path.stat().st_size
    now = utc_now()
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        with session.begin():
            if session.get(AgentRun, run_id) is None:
                target_path.unlink(missing_ok=True)
                raise ArtifactRunNotFoundError(f"Run not found: {run_id}")
            row = RunArtifact(
                id=artifact_id,
                run_id=run_id,
                artifact_type=artifact_type,
                name=safe_name,
                file_path=str(target_path),
                mime_type=mime_type or _infer_mime_type(safe_name),
                size_bytes=size_bytes,
                description=description,
                created_at=now,
            )
            session.add(row)
            session.flush()
            return _build_artifact_record(row)


def save_run_json_artifact(
    run_id: uuid.UUID,
    *,
    name: str,
    payload: dict[str, Any],
    description: str | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> ArtifactRecord:
    return save_run_artifact(
        run_id,
        name=name,
        artifact_type="json",
        content=json.dumps(payload, indent=2, sort_keys=True, default=str),
        mime_type="application/json",
        description=description,
        session_factory=session_factory,
    )


def list_run_artifacts(
    run_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> list[ArtifactRecord] | None:
    session_factory = session_factory or create_session_factory()
    with session_factory() as session:
        if session.get(AgentRun, run_id) is None:
            return None
        rows = session.execute(
            select(RunArtifact)
            .where(RunArtifact.run_id == run_id)
            .order_by(RunArtifact.created_at.desc(), RunArtifact.id.desc())
        ).scalars().all()
    return [_build_artifact_record(row) for row in rows]


def get_artifact(
    artifact_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> ArtifactRecord | None:
    session_factory = session_factory or create_session_factory()
    with session_factory() as session:
        row = session.get(RunArtifact, artifact_id)
        return _build_artifact_record(row) if row is not None else None


def get_run_artifact(
    run_id: uuid.UUID,
    artifact_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> ArtifactRecord | None:
    artifact = get_artifact(artifact_id, session_factory=session_factory)
    if artifact is None or artifact.run_id != run_id:
        return None
    return artifact


def resolve_artifact_file(artifact: ArtifactRecord) -> Path:
    path = Path(artifact.file_path)
    if not path.is_file():
        raise ArtifactFileMissingError(f"Artifact file is missing: {artifact.file_path}")
    return path


def _safe_filename(name: str) -> str:
    base = Path(name.strip() or "artifact").name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return base or "artifact"


def _infer_mime_type(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix in {".txt", ".log", ".md"}:
        return "text/plain"
    return "application/octet-stream"


def _build_artifact_record(row: RunArtifact) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=row.id,
        run_id=row.run_id,
        artifact_type=row.artifact_type,
        name=row.name,
        file_path=row.file_path,
        mime_type=row.mime_type,
        size_bytes=row.size_bytes,
        description=row.description,
        created_at=row.created_at,
    )
