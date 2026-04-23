from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from agentflow.db.models import AgentDefinition, AgentRun, AgentVersion, utc_now
from agentflow.db.session import create_session_factory
from agentflow.services.run_events import (
    RUN_EVENT_RUN_CREATED,
    RUN_EVENT_RUN_ENQUEUED,
    RUN_EVENT_RUN_STARTED,
    RUN_EVENT_WORKER_PICKED_UP_RUN,
    RunEventCreate,
    record_run_events,
)

RunStatus = Literal["pending", "running", "completed", "failed"]

RUN_STATUS_PENDING: RunStatus = "pending"
RUN_STATUS_RUNNING: RunStatus = "running"
RUN_STATUS_COMPLETED: RunStatus = "completed"
RUN_STATUS_FAILED: RunStatus = "failed"
TERMINAL_RUN_STATUSES = frozenset({RUN_STATUS_COMPLETED, RUN_STATUS_FAILED})
_UNSET = object()


@dataclass(frozen=True)
class AgentExecutionTarget:
    agent_id: uuid.UUID
    version_id: uuid.UUID
    version_number: int
    normalized_config_json: dict[str, Any]


@dataclass(frozen=True)
class AgentRunSummary:
    run_id: uuid.UUID
    agent_id: uuid.UUID
    status: RunStatus
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime | None


@dataclass(frozen=True)
class AgentRunDetail:
    run_id: uuid.UUID
    agent_id: uuid.UUID
    version_id: uuid.UUID
    status: RunStatus
    input_json: dict[str, Any] | None
    resolved_config_json: dict[str, Any]
    output_json: dict[str, Any] | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime | None
    updated_at: datetime


def get_agent_execution_target(
    agent_id: uuid.UUID,
    session_factory: sessionmaker[Session] | None = None,
) -> AgentExecutionTarget | None:
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        row = session.execute(
            select(
                AgentDefinition.id.label("agent_id"),
                AgentVersion.id.label("version_id"),
                AgentVersion.version_number.label("version_number"),
                AgentVersion.normalized_config_json.label("normalized_config_json"),
            )
            .join(AgentVersion, AgentVersion.agent_id == AgentDefinition.id)
            .where(AgentDefinition.id == agent_id)
            .order_by(AgentVersion.version_number.desc(), AgentVersion.created_at.desc())
            .limit(1)
        ).one_or_none()

    if row is None:
        return None

    return AgentExecutionTarget(
        agent_id=row.agent_id,
        version_id=row.version_id,
        version_number=row.version_number,
        normalized_config_json=dict(row.normalized_config_json),
    )


def create_agent_run(
    *,
    agent_id: uuid.UUID,
    version_id: uuid.UUID,
    resolved_config_json: dict[str, Any],
    input_json: dict[str, Any] | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> AgentRunDetail:
    session_factory = session_factory or create_session_factory()
    now = utc_now()

    with session_factory() as session:
        with session.begin():
            run = AgentRun(
                agent_id=agent_id,
                version_id=version_id,
                status=RUN_STATUS_PENDING,
                input_json=input_json,
                resolved_config_json=dict(resolved_config_json),
                created_at=now,
                updated_at=now,
            )
            session.add(run)
            session.flush()
            record_run_events(
                run.id,
                events=(
                    RunEventCreate(
                        event_type=RUN_EVENT_RUN_CREATED,
                        message="Run row created.",
                        payload_json={
                            "agent_id": str(agent_id),
                            "version_id": str(version_id),
                            "initial_status": RUN_STATUS_PENDING,
                        },
                    ),
                    RunEventCreate(
                        event_type=RUN_EVENT_RUN_ENQUEUED,
                        message="Run is pending and available for worker pickup.",
                        payload_json={"status": RUN_STATUS_PENDING},
                    ),
                ),
                session=session,
            )

        return _build_run_detail(run)


def claim_next_pending_run(
    *,
    worker_id: str | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> AgentRunDetail | None:
    session_factory = session_factory or create_session_factory()
    now = utc_now()

    with session_factory() as session:
        with session.begin():
            run = session.execute(
                select(AgentRun)
                .where(AgentRun.status == RUN_STATUS_PENDING)
                .order_by(AgentRun.created_at.asc(), AgentRun.id.asc())
                .with_for_update(skip_locked=True)
                .limit(1)
            ).scalars().first()

            if run is None:
                return None

            run.status = RUN_STATUS_RUNNING
            run.started_at = now
            run.updated_at = now
            record_run_events(
                run.id,
                events=(
                    RunEventCreate(
                        event_type=RUN_EVENT_WORKER_PICKED_UP_RUN,
                        message="Worker claimed the pending run.",
                        payload_json={"worker_id": worker_id} if worker_id is not None else None,
                    ),
                    RunEventCreate(
                        event_type=RUN_EVENT_RUN_STARTED,
                        message="Run moved to running state.",
                        payload_json={"status": RUN_STATUS_RUNNING},
                    ),
                ),
                session=session,
            )

        return _build_run_detail(run)


def mark_agent_run_running(
    run_id: uuid.UUID,
    *,
    events: Sequence[RunEventCreate] = (),
    session_factory: sessionmaker[Session] | None = None,
) -> AgentRunDetail | None:
    return _update_agent_run(
        run_id,
        status=RUN_STATUS_RUNNING,
        started_at=utc_now(),
        events=events,
        session_factory=session_factory,
    )


def mark_agent_run_completed(
    run_id: uuid.UUID,
    *,
    output_json: dict[str, Any],
    events: Sequence[RunEventCreate] = (),
    session_factory: sessionmaker[Session] | None = None,
) -> AgentRunDetail | None:
    return _update_agent_run(
        run_id,
        status=RUN_STATUS_COMPLETED,
        output_json=output_json,
        ended_at=utc_now(),
        error_message=None,
        events=events,
        session_factory=session_factory,
    )


def mark_agent_run_failed(
    run_id: uuid.UUID,
    *,
    error_message: str,
    events: Sequence[RunEventCreate] = (),
    session_factory: sessionmaker[Session] | None = None,
) -> AgentRunDetail | None:
    return _update_agent_run(
        run_id,
        status=RUN_STATUS_FAILED,
        error_message=error_message,
        ended_at=utc_now(),
        events=events,
        session_factory=session_factory,
    )


def list_agent_runs(
    session_factory: sessionmaker[Session] | None = None,
) -> list[AgentRunSummary]:
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        rows = session.execute(
            select(
                AgentRun.id,
                AgentRun.agent_id,
                AgentRun.status,
                AgentRun.created_at,
                AgentRun.started_at,
                AgentRun.ended_at,
            )
            .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
        ).all()

    return [
        AgentRunSummary(
            run_id=row.id,
            agent_id=row.agent_id,
            status=row.status,
            created_at=row.created_at,
            started_at=row.started_at,
            ended_at=row.ended_at,
        )
        for row in rows
    ]


def get_agent_run(
    run_id: uuid.UUID,
    session_factory: sessionmaker[Session] | None = None,
) -> AgentRunDetail | None:
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        run = session.get(AgentRun, run_id)

    if run is None:
        return None

    return _build_run_detail(run)


def _update_agent_run(
    run_id: uuid.UUID,
    *,
    status: RunStatus,
    started_at: datetime | None | object = _UNSET,
    ended_at: datetime | None | object = _UNSET,
    output_json: dict[str, Any] | None | object = _UNSET,
    error_message: str | None | object = _UNSET,
    events: Sequence[RunEventCreate] = (),
    session_factory: sessionmaker[Session] | None = None,
) -> AgentRunDetail | None:
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        with session.begin():
            run = session.get(AgentRun, run_id)
            if run is None:
                return None

            run.status = status
            run.updated_at = utc_now()

            if started_at is not _UNSET:
                run.started_at = started_at
            if ended_at is not _UNSET:
                run.ended_at = ended_at
            if output_json is not _UNSET:
                run.output_json = output_json
            if error_message is not _UNSET:
                run.error_message = error_message
            if events:
                record_run_events(run.id, events=events, session=session)

        return _build_run_detail(run)


def _build_run_detail(run: AgentRun) -> AgentRunDetail:
    return AgentRunDetail(
        run_id=run.id,
        agent_id=run.agent_id,
        version_id=run.version_id,
        status=run.status,
        input_json=run.input_json,
        resolved_config_json=run.resolved_config_json,
        output_json=run.output_json,
        error_message=run.error_message,
        created_at=run.created_at,
        started_at=run.started_at,
        ended_at=run.ended_at,
        updated_at=run.updated_at,
    )
