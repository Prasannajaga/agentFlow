from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from agentflow.db.models import AgentRun, AgentVersion, RunLabel, VersionLabel, utc_now
from agentflow.db.session import create_session_factory


class LabelError(RuntimeError):
    """Base error for label operations."""


class LabelInvalidError(LabelError):
    pass


class LabelDuplicateError(LabelError):
    pass


class LabelTargetNotFoundError(LabelError):
    pass


@dataclass(frozen=True)
class LabelRecord:
    label: str
    created_at: datetime


def normalize_label(label: str) -> str:
    normalized = " ".join(label.strip().lower().split())
    if not normalized:
        raise LabelInvalidError("Label cannot be empty.")
    if len(normalized) > 64:
        raise LabelInvalidError("Label cannot be longer than 64 characters.")
    if "/" in normalized:
        raise LabelInvalidError("Label cannot contain '/'.")
    return normalized


def add_run_label(
    run_id: uuid.UUID,
    label: str,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> LabelRecord:
    normalized = normalize_label(label)
    session_factory = session_factory or create_session_factory()
    now = utc_now()

    with session_factory() as session:
        with session.begin():
            if session.get(AgentRun, run_id) is None:
                raise LabelTargetNotFoundError(f"Run not found: {run_id}")
            record = RunLabel(run_id=run_id, label=normalized, created_at=now)
            session.add(record)
            try:
                session.flush()
            except IntegrityError as exc:
                raise LabelDuplicateError(f"Run already has label: {normalized}") from exc

    return LabelRecord(label=normalized, created_at=now)


def remove_run_label(
    run_id: uuid.UUID,
    label: str,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> bool:
    normalized = normalize_label(label)
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        with session.begin():
            if session.get(AgentRun, run_id) is None:
                raise LabelTargetNotFoundError(f"Run not found: {run_id}")
            result = session.execute(
                delete(RunLabel).where(RunLabel.run_id == run_id, RunLabel.label == normalized)
            )
    return bool(result.rowcount)


def list_run_labels(
    run_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> list[LabelRecord]:
    session_factory = session_factory or create_session_factory()
    with session_factory() as session:
        rows = session.execute(
            select(RunLabel.label, RunLabel.created_at)
            .where(RunLabel.run_id == run_id)
            .order_by(RunLabel.label.asc())
        ).all()
    return [LabelRecord(label=row.label, created_at=row.created_at) for row in rows]


def list_run_labels_for_runs(
    run_ids: list[uuid.UUID],
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> dict[uuid.UUID, list[str]]:
    if not run_ids:
        return {}
    session_factory = session_factory or create_session_factory()
    with session_factory() as session:
        rows = session.execute(
            select(RunLabel.run_id, RunLabel.label)
            .where(RunLabel.run_id.in_(run_ids))
            .order_by(RunLabel.label.asc())
        ).all()
    labels: dict[uuid.UUID, list[str]] = {run_id: [] for run_id in run_ids}
    for row in rows:
        labels.setdefault(row.run_id, []).append(row.label)
    return labels


def add_version_label(
    version_id: uuid.UUID,
    label: str,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> LabelRecord:
    normalized = normalize_label(label)
    session_factory = session_factory or create_session_factory()
    now = utc_now()

    with session_factory() as session:
        with session.begin():
            if session.get(AgentVersion, version_id) is None:
                raise LabelTargetNotFoundError(f"Version not found: {version_id}")
            record = VersionLabel(version_id=version_id, label=normalized, created_at=now)
            session.add(record)
            try:
                session.flush()
            except IntegrityError as exc:
                raise LabelDuplicateError(f"Version already has label: {normalized}") from exc

    return LabelRecord(label=normalized, created_at=now)


def remove_version_label(
    version_id: uuid.UUID,
    label: str,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> bool:
    normalized = normalize_label(label)
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        with session.begin():
            if session.get(AgentVersion, version_id) is None:
                raise LabelTargetNotFoundError(f"Version not found: {version_id}")
            result = session.execute(
                delete(VersionLabel).where(VersionLabel.version_id == version_id, VersionLabel.label == normalized)
            )
    return bool(result.rowcount)


def list_version_labels(
    version_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> list[LabelRecord]:
    session_factory = session_factory or create_session_factory()
    with session_factory() as session:
        rows = session.execute(
            select(VersionLabel.label, VersionLabel.created_at)
            .where(VersionLabel.version_id == version_id)
            .order_by(VersionLabel.label.asc())
        ).all()
    return [LabelRecord(label=row.label, created_at=row.created_at) for row in rows]


def list_version_labels_for_versions(
    version_ids: list[uuid.UUID],
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> dict[uuid.UUID, list[str]]:
    if not version_ids:
        return {}
    session_factory = session_factory or create_session_factory()
    with session_factory() as session:
        rows = session.execute(
            select(VersionLabel.version_id, VersionLabel.label)
            .where(VersionLabel.version_id.in_(version_ids))
            .order_by(VersionLabel.label.asc())
        ).all()
    labels: dict[uuid.UUID, list[str]] = {version_id: [] for version_id in version_ids}
    for row in rows:
        labels.setdefault(row.version_id, []).append(row.label)
    return labels
