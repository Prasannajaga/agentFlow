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
    RUN_EVENT_RUN_ATTEMPT_FAILED,
    RUN_EVENT_RUN_ATTEMPT_STARTED,
    RUN_EVENT_RUN_CREATED_WITH_VERSION,
    RUN_EVENT_RUN_CREATED,
    RUN_EVENT_RUN_ENQUEUED,
    RUN_EVENT_RUN_FAILED,
    RUN_EVENT_RUN_RERUN_CREATED,
    RUN_EVENT_RUN_RERUN_REQUESTED,
    RUN_EVENT_RUN_RETRY_LIMIT_REACHED,
    RUN_EVENT_RUN_RETRY_SCHEDULED,
    RUN_EVENT_RUN_STARTED,
    RUN_EVENT_RUN_CLAIMED_BY_WORKER,
    RUN_EVENT_WORKER_PICKED_UP_RUN,
    RunEventCreate,
    record_run_events,
)
from agentflow.services.retry_policy import extract_max_attempts, should_retry_failed_attempt
from agentflow.services.artifact_service import save_run_json_artifact
from agentflow.services.worker_ops import clear_run_claim, mark_run_claimed

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
    version_id: uuid.UUID
    source_run_id: uuid.UUID | None
    status: RunStatus
    attempt_count: int
    max_attempts: int
    retryable: bool
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime | None
    claimed_by_worker: str | None = None
    claimed_at: datetime | None = None


@dataclass(frozen=True)
class AgentRunDetail:
    run_id: uuid.UUID
    agent_id: uuid.UUID
    version_id: uuid.UUID
    source_run_id: uuid.UUID | None
    status: RunStatus
    input_json: dict[str, Any] | None
    resolved_config_json: dict[str, Any]
    output_json: dict[str, Any] | None
    error_message: str | None
    last_error_type: str | None
    attempt_count: int
    max_attempts: int
    retryable: bool
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime | None
    updated_at: datetime
    claimed_by_worker: str | None = None
    claimed_at: datetime | None = None


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


def get_agent_version_execution_target(
    version_id: uuid.UUID,
    session_factory: sessionmaker[Session] | None = None,
) -> AgentExecutionTarget | None:
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        row = session.execute(
            select(
                AgentVersion.agent_id.label("agent_id"),
                AgentVersion.id.label("version_id"),
                AgentVersion.version_number.label("version_number"),
                AgentVersion.normalized_config_json.label("normalized_config_json"),
            )
            .where(AgentVersion.id == version_id)
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
    source_run_id: uuid.UUID | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> AgentRunDetail:
    session_factory = session_factory or create_session_factory()
    now = utc_now()
    snapshot = _validate_resolved_config_snapshot(resolved_config_json)

    with session_factory() as session:
        with session.begin():
            run = _create_agent_run_in_session(
                session,
                agent_id=agent_id,
                version_id=version_id,
                resolved_config_json=snapshot,
                input_json=input_json,
                source_run_id=source_run_id,
                created_at=now,
            )

        return _build_run_detail(run)


def create_rerun_from_run(
    source_run_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> AgentRunDetail | None:
    session_factory = session_factory or create_session_factory()
    now = utc_now()

    with session_factory() as session:
        with session.begin():
            source_run = session.get(AgentRun, source_run_id)
            if source_run is None:
                return None

            snapshot = _validate_resolved_config_snapshot(source_run.resolved_config_json)
            run = _create_agent_run_in_session(
                session,
                agent_id=source_run.agent_id,
                version_id=source_run.version_id,
                resolved_config_json=snapshot,
                input_json=dict(source_run.input_json) if source_run.input_json is not None else None,
                source_run_id=source_run.id,
                created_at=now,
            )
            record_run_events(
                source_run.id,
                events=(
                    RunEventCreate(
                        event_type=RUN_EVENT_RUN_RERUN_REQUESTED,
                        message="Rerun requested from this run snapshot.",
                        payload_json={
                            "source_run_id": str(source_run.id),
                            "new_run_id": str(run.id),
                        },
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

            _start_run_attempt(run, started_at=now)
            mark_run_claimed(run, worker_name=worker_id, claimed_at=now)
            record_run_events(
                run.id,
                events=(
                    RunEventCreate(
                        event_type=RUN_EVENT_WORKER_PICKED_UP_RUN,
                        message="Worker claimed the pending run.",
                        payload_json={"worker_id": worker_id} if worker_id is not None else None,
                    ),
                    RunEventCreate(
                        event_type=RUN_EVENT_RUN_CLAIMED_BY_WORKER,
                        message="Run claim metadata recorded for stale-run recovery.",
                        payload_json={
                            "worker_name": worker_id,
                            "claimed_at": now.isoformat(),
                        },
                    ),
                    RunEventCreate(
                        event_type=RUN_EVENT_RUN_ATTEMPT_STARTED,
                        message="Run execution attempt started.",
                        payload_json={
                            "attempt_count": run.attempt_count,
                            "max_attempts": run.max_attempts,
                        },
                    ),
                    RunEventCreate(
                        event_type=RUN_EVENT_RUN_STARTED,
                        message="Run moved to running state.",
                        payload_json={
                            "status": RUN_STATUS_RUNNING,
                            "attempt_count": run.attempt_count,
                            "max_attempts": run.max_attempts,
                        },
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
        increment_attempt=True,
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
    completed_run = _update_agent_run(
        run_id,
        status=RUN_STATUS_COMPLETED,
        output_json=output_json,
        ended_at=utc_now(),
        error_message=None,
        last_error_type=None,
        events=events,
        session_factory=session_factory,
    )
    if completed_run is not None:
        save_run_json_artifact(
            completed_run.run_id,
            name="output.json",
            payload=output_json,
            description="Full run output JSON.",
            session_factory=session_factory,
        )
    return completed_run


def mark_agent_run_failed(
    run_id: uuid.UUID,
    *,
    error_message: str,
    last_error_type: str | None = None,
    output_json: dict[str, Any] | None | object = _UNSET,
    events: Sequence[RunEventCreate] = (),
    session_factory: sessionmaker[Session] | None = None,
) -> AgentRunDetail | None:
    session_factory = session_factory or create_session_factory()
    now = utc_now()

    with session_factory() as session:
        with session.begin():
            run = session.get(AgentRun, run_id)
            if run is None:
                return None

            run.error_message = error_message
            run.last_error_type = last_error_type
            if output_json is not _UNSET:
                run.output_json = output_json
            run.updated_at = now

            attempt_payload = {
                "attempt_count": run.attempt_count,
                "max_attempts": run.max_attempts,
            }
            if last_error_type is not None:
                attempt_payload["error_type"] = last_error_type

            retry_events: list[RunEventCreate] = [
                *events,
                RunEventCreate(
                    event_type=RUN_EVENT_RUN_ATTEMPT_FAILED,
                    message="Run execution attempt failed.",
                    payload_json=attempt_payload,
                ),
            ]

            if should_retry_failed_attempt(
                attempt_count=run.attempt_count,
                max_attempts=run.max_attempts,
                retryable=run.retryable,
            ):
                run.status = RUN_STATUS_PENDING
                run.started_at = None
                run.ended_at = None
                clear_run_claim(run)
                retry_events.append(
                    RunEventCreate(
                        event_type=RUN_EVENT_RUN_RETRY_SCHEDULED,
                        message="Run will be retried by the worker.",
                        payload_json={
                            "attempt_count": run.attempt_count,
                            "max_attempts": run.max_attempts,
                            "next_status": RUN_STATUS_PENDING,
                        },
                    )
                )
            else:
                run.status = RUN_STATUS_FAILED
                run.ended_at = now
                if run.retryable and run.attempt_count >= run.max_attempts:
                    retry_events.append(
                        RunEventCreate(
                            event_type=RUN_EVENT_RUN_RETRY_LIMIT_REACHED,
                            message="Run retry limit reached.",
                            payload_json={
                                "attempt_count": run.attempt_count,
                                "max_attempts": run.max_attempts,
                            },
                        )
                    )
                retry_events.append(
                    RunEventCreate(
                        event_type=RUN_EVENT_RUN_FAILED,
                        message="Run failed during execution.",
                        payload_json={
                            "status": RUN_STATUS_FAILED,
                            "attempt_count": run.attempt_count,
                            "max_attempts": run.max_attempts,
                            "error_message": error_message,
                        },
                    )
                )

            record_run_events(run.id, events=retry_events, session=session)

        return _build_run_detail(run)


def list_agent_runs(
    *,
    limit: int | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> list[AgentRunSummary]:
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        statement = (
            select(
                AgentRun.id,
                AgentRun.agent_id,
                AgentRun.version_id,
                AgentRun.source_run_id,
                AgentRun.status,
                AgentRun.attempt_count,
                AgentRun.max_attempts,
                AgentRun.retryable,
                AgentRun.claimed_by_worker,
                AgentRun.claimed_at,
                AgentRun.created_at,
                AgentRun.started_at,
                AgentRun.ended_at,
            )
            .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
        )
        if limit is not None:
            statement = statement.limit(limit)

        rows = session.execute(statement).all()

    return [
        AgentRunSummary(
            run_id=row.id,
            agent_id=row.agent_id,
            version_id=row.version_id,
            source_run_id=row.source_run_id,
            status=row.status,
            attempt_count=row.attempt_count,
            max_attempts=row.max_attempts,
            retryable=row.retryable,
            claimed_by_worker=row.claimed_by_worker,
            claimed_at=row.claimed_at,
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
    last_error_type: str | None | object = _UNSET,
    increment_attempt: bool = False,
    events: Sequence[RunEventCreate] = (),
    session_factory: sessionmaker[Session] | None = None,
) -> AgentRunDetail | None:
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        with session.begin():
            run = session.get(AgentRun, run_id)
            if run is None:
                return None

            if increment_attempt:
                _start_run_attempt(run, started_at=started_at if isinstance(started_at, datetime) else utc_now())
            else:
                run.status = status
                if status == RUN_STATUS_PENDING:
                    clear_run_claim(run)
            run.updated_at = utc_now()

            if started_at is not _UNSET and not increment_attempt:
                run.started_at = started_at
            if ended_at is not _UNSET:
                run.ended_at = ended_at
            if output_json is not _UNSET:
                run.output_json = output_json
            if error_message is not _UNSET:
                run.error_message = error_message
            if last_error_type is not _UNSET:
                run.last_error_type = last_error_type
            if events or increment_attempt:
                persisted_events: list[RunEventCreate] = []
                if increment_attempt:
                    persisted_events.append(
                        RunEventCreate(
                            event_type=RUN_EVENT_RUN_ATTEMPT_STARTED,
                            message="Run execution attempt started.",
                            payload_json={
                                "attempt_count": run.attempt_count,
                                "max_attempts": run.max_attempts,
                            },
                        )
                    )
                persisted_events.extend(events)
                record_run_events(run.id, events=persisted_events, session=session)

        return _build_run_detail(run)


def _build_run_detail(run: AgentRun) -> AgentRunDetail:
    return AgentRunDetail(
        run_id=run.id,
        agent_id=run.agent_id,
        version_id=run.version_id,
        source_run_id=run.source_run_id,
        status=run.status,
        input_json=run.input_json,
        resolved_config_json=run.resolved_config_json,
        output_json=run.output_json,
        error_message=run.error_message,
        last_error_type=run.last_error_type,
        attempt_count=run.attempt_count,
        max_attempts=run.max_attempts,
        retryable=run.retryable,
        claimed_by_worker=run.claimed_by_worker,
        claimed_at=run.claimed_at,
        created_at=run.created_at,
        started_at=run.started_at,
        ended_at=run.ended_at,
        updated_at=run.updated_at,
    )


def _create_agent_run_in_session(
    session: Session,
    *,
    agent_id: uuid.UUID,
    version_id: uuid.UUID,
    resolved_config_json: dict[str, Any],
    input_json: dict[str, Any] | None,
    source_run_id: uuid.UUID | None = None,
    created_at: datetime,
    extra_events: Sequence[RunEventCreate] = (),
) -> AgentRun:
    max_attempts = extract_max_attempts(resolved_config_json)
    run = AgentRun(
        agent_id=agent_id,
        version_id=version_id,
        source_run_id=source_run_id,
        status=RUN_STATUS_PENDING,
        input_json=input_json,
        resolved_config_json=dict(resolved_config_json),
        attempt_count=0,
        max_attempts=max_attempts,
        retryable=True,
        created_at=created_at,
        updated_at=created_at,
    )
    session.add(run)
    session.flush()

    events: list[RunEventCreate] = [
        RunEventCreate(
            event_type=RUN_EVENT_RUN_CREATED,
            message="Run row created.",
            payload_json={
                "agent_id": str(agent_id),
                "version_id": str(version_id),
                "source_run_id": str(source_run_id) if source_run_id is not None else None,
                "initial_status": RUN_STATUS_PENDING,
                "attempt_count": 0,
                "max_attempts": max_attempts,
                "retryable": True,
            },
        ),
        RunEventCreate(
            event_type=RUN_EVENT_RUN_CREATED_WITH_VERSION,
            message="Run pinned to an immutable agent version snapshot.",
            payload_json={
                "agent_id": str(agent_id),
                "version_id": str(version_id),
            },
        ),
    ]
    if source_run_id is not None:
        events.append(
            RunEventCreate(
                event_type=RUN_EVENT_RUN_RERUN_CREATED,
                message="Rerun created from pinned source run snapshot.",
                payload_json={
                    "source_run_id": str(source_run_id),
                    "new_run_id": str(run.id),
                    "version_id": str(version_id),
                },
            )
        )
    events.extend(extra_events)
    events.append(
        RunEventCreate(
            event_type=RUN_EVENT_RUN_ENQUEUED,
            message="Run is pending and available for worker pickup.",
            payload_json={"status": RUN_STATUS_PENDING, "max_attempts": max_attempts},
        )
    )
    record_run_events(run.id, events=events, session=session)
    return run


def _validate_resolved_config_snapshot(resolved_config_json: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(resolved_config_json, dict) or not resolved_config_json:
        raise ValueError("Run snapshot is missing resolved_config_json.")

    return dict(resolved_config_json)


def _start_run_attempt(run: AgentRun, *, started_at: datetime) -> None:
    run.status = RUN_STATUS_RUNNING
    run.started_at = started_at
    run.ended_at = None
    run.attempt_count += 1
    run.updated_at = started_at
