from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from agentflow.db.models import AgentRun, RunEvent, utc_now
from agentflow.db.session import create_session_factory

RUN_EVENT_RUN_CREATED = "RUN_CREATED"
RUN_EVENT_RUN_ENQUEUED = "RUN_ENQUEUED"
RUN_EVENT_WORKER_PICKED_UP_RUN = "WORKER_PICKED_UP_RUN"
RUN_EVENT_RUN_STARTED = "RUN_STARTED"
RUN_EVENT_PROVIDER_EXECUTION_STARTED = "PROVIDER_EXECUTION_STARTED"
RUN_EVENT_PROVIDER_REQUEST_PREPARED = "PROVIDER_REQUEST_PREPARED"
RUN_EVENT_PROVIDER_EXECUTION_COMPLETED = "PROVIDER_EXECUTION_COMPLETED"
RUN_EVENT_PROVIDER_EXECUTION_FAILED = "PROVIDER_EXECUTION_FAILED"
RUN_EVENT_RUN_COMPLETED = "RUN_COMPLETED"
RUN_EVENT_RUN_FAILED = "RUN_FAILED"


@dataclass(frozen=True)
class RunEventCreate:
    event_type: str
    message: str | None = None
    payload_json: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunEventRecord:
    event_id: uuid.UUID
    run_id: uuid.UUID
    event_type: str
    message: str | None
    payload_json: dict[str, Any] | None
    created_at: datetime


@dataclass(frozen=True)
class RunEventSummary:
    event_count: int
    latest_event_type: str | None
    latest_created_at: datetime | None


def record_run_event(
    run_id: uuid.UUID,
    *,
    event_type: str,
    message: str | None = None,
    payload_json: dict[str, Any] | None = None,
    created_at: datetime | None = None,
    session: Session | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> RunEventRecord:
    created_event = RunEventCreate(
        event_type=event_type,
        message=message,
        payload_json=payload_json,
    )
    return record_run_events(
        run_id,
        events=(created_event,),
        created_at=created_at,
        session=session,
        session_factory=session_factory,
    )[0]


def record_run_events(
    run_id: uuid.UUID,
    *,
    events: Sequence[RunEventCreate],
    created_at: datetime | None = None,
    session: Session | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> tuple[RunEventRecord, ...]:
    if not events:
        return ()

    if session is not None:
        return _record_run_events_in_session(
            session,
            run_id,
            events=events,
            created_at=created_at,
        )

    resolved_session_factory = session_factory or create_session_factory()
    with resolved_session_factory() as owned_session:
        with owned_session.begin():
            return _record_run_events_in_session(
                owned_session,
                run_id,
                events=events,
                created_at=created_at,
            )


def list_run_events(
    run_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> list[RunEventRecord]:
    resolved_session_factory = session_factory or create_session_factory()

    with resolved_session_factory() as session:
        rows = session.execute(
            select(RunEvent)
            .where(RunEvent.run_id == run_id)
            .order_by(RunEvent.created_at.asc(), RunEvent.id.asc())
        ).scalars().all()

    return [_build_run_event_record(row) for row in rows]


def get_run_event_summary(
    run_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> RunEventSummary:
    resolved_session_factory = session_factory or create_session_factory()

    with resolved_session_factory() as session:
        event_count = session.execute(
            select(func.count())
            .select_from(RunEvent)
            .where(RunEvent.run_id == run_id)
        ).scalar_one()

        latest_event = session.execute(
            select(RunEvent.event_type, RunEvent.created_at)
            .where(RunEvent.run_id == run_id)
            .order_by(RunEvent.created_at.desc(), RunEvent.id.desc())
            .limit(1)
        ).one_or_none()

    return RunEventSummary(
        event_count=int(event_count),
        latest_event_type=latest_event.event_type if latest_event is not None else None,
        latest_created_at=latest_event.created_at if latest_event is not None else None,
    )


def _record_run_events_in_session(
    session: Session,
    run_id: uuid.UUID,
    *,
    events: Sequence[RunEventCreate],
    created_at: datetime | None = None,
) -> tuple[RunEventRecord, ...]:
    if session.get(AgentRun, run_id) is None:
        raise ValueError(f"Run not found: {run_id}")

    persisted_events: list[RunEvent] = []
    for event in events:
        event_row = RunEvent(
            run_id=run_id,
            event_type=event.event_type,
            message=event.message,
            payload_json=dict(event.payload_json) if event.payload_json is not None else None,
            created_at=created_at or utc_now(),
        )
        session.add(event_row)
        persisted_events.append(event_row)

    session.flush()
    return tuple(_build_run_event_record(event_row) for event_row in persisted_events)


def _build_run_event_record(event: RunEvent) -> RunEventRecord:
    return RunEventRecord(
        event_id=event.id,
        run_id=event.run_id,
        event_type=event.event_type,
        message=event.message,
        payload_json=dict(event.payload_json) if event.payload_json is not None else None,
        created_at=event.created_at,
    )
