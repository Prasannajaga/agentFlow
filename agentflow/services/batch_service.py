from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from agentflow.db.models import AgentDefinition, AgentInputPreset, AgentRun, AgentVersion, RunBatch, RunBatchItem, utc_now
from agentflow.db.session import create_session_factory
from agentflow.services.run_queries import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_PENDING,
    RUN_STATUS_RUNNING,
    _create_agent_run_in_session,
)

BATCH_STATUS_PENDING = "pending"
BATCH_STATUS_RUNNING = "running"
BATCH_STATUS_COMPLETED = "completed"
BATCH_STATUS_FAILED = "failed"
BATCH_STATUS_PARTIAL = "partial"


class BatchError(RuntimeError):
    """Base error for batch operations."""


class BatchInvalidError(BatchError):
    pass


class BatchNotFoundError(BatchError):
    pass


class BatchAgentNotFoundError(BatchError):
    pass


class BatchPresetNotFoundError(BatchError):
    pass


class BatchPresetAgentMismatchError(BatchError):
    pass


class BatchVersionNotFoundError(BatchError):
    pass


class BatchVersionAgentMismatchError(BatchError):
    pass


@dataclass(frozen=True)
class BatchSummary:
    batch_id: uuid.UUID
    agent_id: uuid.UUID
    version_id: uuid.UUID
    name: str | None
    status: str
    item_count: int
    pending_count: int
    running_count: int
    completed_count: int
    failed_count: int
    first_started_at: datetime | None
    last_ended_at: datetime | None
    elapsed_seconds: float | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class BatchItemRecord:
    item_id: uuid.UUID
    batch_id: uuid.UUID
    run_id: uuid.UUID
    preset_id: uuid.UUID | None
    preset_name: str | None
    status: str
    attempt_count: int
    max_attempts: int
    input_json: dict[str, Any] | None
    result_preview: str | None
    started_at: datetime | None
    ended_at: datetime | None
    created_at: datetime


@dataclass(frozen=True)
class BatchDetail:
    summary: BatchSummary
    items: list[BatchItemRecord]


def create_batch_from_presets(
    agent_id: uuid.UUID,
    *,
    preset_ids: list[uuid.UUID],
    version_id: uuid.UUID | None = None,
    name: str | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> BatchDetail:
    ordered_preset_ids = _normalize_preset_ids(preset_ids)
    normalized_name = name.strip() if name and name.strip() else None
    if normalized_name is not None and len(normalized_name) > 120:
        raise BatchInvalidError("Batch name cannot be longer than 120 characters.")

    session_factory = session_factory or create_session_factory()
    now = utc_now()

    with session_factory() as session:
        with session.begin():
            if session.get(AgentDefinition, agent_id) is None:
                raise BatchAgentNotFoundError(f"Agent not found: {agent_id}")

            presets = _load_presets(session, ordered_preset_ids)
            if len(presets) != len(ordered_preset_ids):
                found_ids = {preset.id for preset in presets}
                missing_ids = [preset_id for preset_id in ordered_preset_ids if preset_id not in found_ids]
                raise BatchPresetNotFoundError(f"Preset not found: {missing_ids[0]}")

            for preset in presets:
                if preset.agent_id != agent_id:
                    raise BatchPresetAgentMismatchError(
                        f"Preset {preset.id} belongs to agent {preset.agent_id}, not requested agent {agent_id}."
                    )

            version = _resolve_batch_version(session, agent_id=agent_id, version_id=version_id)
            batch = RunBatch(
                agent_id=agent_id,
                version_id=version.id,
                name=normalized_name,
                status=BATCH_STATUS_PENDING,
                created_at=now,
                updated_at=now,
            )
            session.add(batch)
            session.flush()

            presets_by_id = {preset.id: preset for preset in presets}
            for preset_id in ordered_preset_ids:
                preset = presets_by_id[preset_id]
                run = _create_agent_run_in_session(
                    session,
                    agent_id=agent_id,
                    version_id=version.id,
                    resolved_config_json=dict(version.normalized_config_json),
                    input_json=dict(preset.input_json),
                    created_at=now,
                )
                session.add(
                    RunBatchItem(
                        batch_id=batch.id,
                        run_id=run.id,
                        preset_id=preset.id,
                        created_at=now,
                    )
                )

            batch_id = batch.id

    detail = get_batch(batch_id, session_factory=session_factory)
    if detail is None:
        raise BatchNotFoundError(f"Batch not found after creation: {batch_id}")
    return detail


def list_batches(
    agent_id: uuid.UUID | None = None,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> list[BatchSummary]:
    session_factory = session_factory or create_session_factory()
    with session_factory() as session:
        statement = select(RunBatch).order_by(RunBatch.created_at.desc(), RunBatch.id.desc())
        if agent_id is not None:
            statement = statement.where(RunBatch.agent_id == agent_id)
        batches = session.execute(statement).scalars().all()
        return [_build_batch_summary(session, batch) for batch in batches]


def get_batch(
    batch_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> BatchDetail | None:
    session_factory = session_factory or create_session_factory()
    with session_factory() as session:
        batch = session.get(RunBatch, batch_id)
        if batch is None:
            return None
        summary = _build_batch_summary(session, batch)
        items = _list_batch_items(session, batch_id)
    return BatchDetail(summary=summary, items=items)


def _normalize_preset_ids(preset_ids: list[uuid.UUID]) -> list[uuid.UUID]:
    if not preset_ids:
        raise BatchInvalidError("At least one preset id is required.")
    seen: set[uuid.UUID] = set()
    ordered: list[uuid.UUID] = []
    for preset_id in preset_ids:
        if preset_id in seen:
            raise BatchInvalidError(f"Duplicate preset id: {preset_id}")
        seen.add(preset_id)
        ordered.append(preset_id)
    return ordered


def _load_presets(session: Session, preset_ids: list[uuid.UUID]) -> list[AgentInputPreset]:
    rows = session.execute(
        select(AgentInputPreset).where(AgentInputPreset.id.in_(preset_ids))
    ).scalars().all()
    presets_by_id = {preset.id: preset for preset in rows}
    return [presets_by_id[preset_id] for preset_id in preset_ids if preset_id in presets_by_id]


def _resolve_batch_version(session: Session, *, agent_id: uuid.UUID, version_id: uuid.UUID | None) -> AgentVersion:
    if version_id is not None:
        version = session.get(AgentVersion, version_id)
        if version is None:
            raise BatchVersionNotFoundError(f"Agent version not found: {version_id}")
        if version.agent_id != agent_id:
            raise BatchVersionAgentMismatchError(
                f"Agent version {version_id} belongs to agent {version.agent_id}, not requested agent {agent_id}."
            )
        return version

    version = session.execute(
        select(AgentVersion)
        .where(AgentVersion.agent_id == agent_id)
        .order_by(AgentVersion.version_number.desc(), AgentVersion.created_at.desc())
        .limit(1)
    ).scalars().first()
    if version is None:
        raise BatchVersionNotFoundError(f"No versions found for agent: {agent_id}")
    return version


def _build_batch_summary(session: Session, batch: RunBatch) -> BatchSummary:
    rows = session.execute(
        select(AgentRun.status, AgentRun.started_at, AgentRun.ended_at)
        .join(RunBatchItem, RunBatchItem.run_id == AgentRun.id)
        .where(RunBatchItem.batch_id == batch.id)
    ).all()
    statuses = [row.status for row in rows]
    started_values = [row.started_at for row in rows if row.started_at is not None]
    ended_values = [row.ended_at for row in rows if row.ended_at is not None]
    first_started_at = min(started_values) if started_values else None
    last_ended_at = max(ended_values) if ended_values else None
    elapsed_seconds = (
        (last_ended_at - first_started_at).total_seconds()
        if first_started_at is not None and last_ended_at is not None
        else None
    )
    return BatchSummary(
        batch_id=batch.id,
        agent_id=batch.agent_id,
        version_id=batch.version_id,
        name=batch.name,
        status=_derive_batch_status(statuses),
        item_count=len(statuses),
        pending_count=_count_status(statuses, RUN_STATUS_PENDING),
        running_count=_count_status(statuses, RUN_STATUS_RUNNING),
        completed_count=_count_status(statuses, RUN_STATUS_COMPLETED),
        failed_count=_count_status(statuses, RUN_STATUS_FAILED),
        first_started_at=first_started_at,
        last_ended_at=last_ended_at,
        elapsed_seconds=elapsed_seconds,
        created_at=batch.created_at,
        updated_at=batch.updated_at,
    )


def _list_batch_items(session: Session, batch_id: uuid.UUID) -> list[BatchItemRecord]:
    rows = session.execute(
        select(
            RunBatchItem.id.label("item_id"),
            RunBatchItem.batch_id,
            RunBatchItem.run_id,
            RunBatchItem.preset_id,
            AgentInputPreset.name.label("preset_name"),
            AgentRun.status,
            AgentRun.attempt_count,
            AgentRun.max_attempts,
            AgentRun.input_json,
            AgentRun.output_json,
            AgentRun.error_message,
            AgentRun.started_at,
            AgentRun.ended_at,
            AgentRun.created_at,
        )
        .join(AgentRun, AgentRun.id == RunBatchItem.run_id)
        .outerjoin(AgentInputPreset, AgentInputPreset.id == RunBatchItem.preset_id)
        .where(RunBatchItem.batch_id == batch_id)
        .order_by(RunBatchItem.created_at.asc(), RunBatchItem.id.asc())
    ).all()
    return [
        BatchItemRecord(
            item_id=row.item_id,
            batch_id=row.batch_id,
            run_id=row.run_id,
            preset_id=row.preset_id,
            preset_name=row.preset_name,
            status=row.status,
            attempt_count=row.attempt_count,
            max_attempts=row.max_attempts,
            input_json=dict(row.input_json) if row.input_json is not None else None,
            result_preview=_build_result_preview(
                output_json=dict(row.output_json) if row.output_json is not None else None,
                error_message=row.error_message,
            ),
            started_at=row.started_at,
            ended_at=row.ended_at,
            created_at=row.created_at,
        )
        for row in rows
    ]


def _derive_batch_status(statuses: list[str]) -> str:
    if not statuses:
        return BATCH_STATUS_PENDING
    unique_statuses = set(statuses)
    if unique_statuses == {RUN_STATUS_PENDING}:
        return BATCH_STATUS_PENDING
    if RUN_STATUS_RUNNING in unique_statuses:
        return BATCH_STATUS_RUNNING
    if unique_statuses == {RUN_STATUS_COMPLETED}:
        return BATCH_STATUS_COMPLETED
    if unique_statuses == {RUN_STATUS_FAILED}:
        return BATCH_STATUS_FAILED
    return BATCH_STATUS_PARTIAL


def _count_status(statuses: list[str], status: str) -> int:
    return sum(1 for value in statuses if value == status)


def _build_result_preview(*, output_json: dict[str, Any] | None, error_message: str | None) -> str | None:
    if output_json:
        output_text = output_json.get("output_text")
        if output_text:
            return _truncate_preview(str(output_text))
        return _truncate_preview(json.dumps(output_json, sort_keys=True, default=str))
    if error_message:
        return _truncate_preview(error_message)
    return None


def _truncate_preview(value: str, *, width: int = 100) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= width:
        return collapsed
    return f"{collapsed[: max(width - 3, 0)].rstrip()}..."
