from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from agentflow.db.models import AgentRun, WorkerHeartbeat, utc_now
from agentflow.db.session import create_session_factory
from agentflow.services.run_events import (
    RUN_EVENT_RUN_MARKED_STALE,
    RUN_EVENT_RUN_RECOVERED_TO_PENDING,
    RunEventCreate,
    record_run_events,
)

RUN_STATUS_PENDING = "pending"
RUN_STATUS_RUNNING = "running"

WORKER_STATUS_STARTING = "starting"
WORKER_STATUS_IDLE = "idle"
WORKER_STATUS_PROCESSING = "processing"
WORKER_STATUS_STOPPED = "stopped"

DEFAULT_MAX_RUNNING_AGE_MULTIPLIER = 6


@dataclass(frozen=True)
class WorkerIdentity:
    worker_name: str
    host: str | None
    pid: int | None


@dataclass(frozen=True)
class WorkerStatusRecord:
    worker_id: uuid.UUID
    worker_name: str
    host: str | None
    pid: int | None
    status: str
    last_heartbeat_at: datetime
    started_at: datetime
    updated_at: datetime
    metadata_json: dict[str, Any] | None
    heartbeat_age_seconds: int
    freshness: str


@dataclass(frozen=True)
class StaleRunCandidate:
    run_id: uuid.UUID
    agent_id: uuid.UUID
    claimed_by_worker: str | None
    claimed_at: datetime | None
    started_at: datetime | None
    updated_at: datetime
    worker_last_heartbeat_at: datetime | None
    worker_status: str | None
    stale_age_seconds: int
    reason: str


@dataclass(frozen=True)
class RecoverStaleRunsResult:
    candidate_count: int
    recovered_count: int
    skipped_count: int
    stale_runs: tuple[StaleRunCandidate, ...]


def heartbeat_worker(
    identity: WorkerIdentity,
    *,
    status: str,
    metadata_json: dict[str, Any] | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> WorkerStatusRecord:
    resolved_session_factory = session_factory or create_session_factory()
    now = utc_now()

    with resolved_session_factory() as session:
        with session.begin():
            existing = session.execute(
                select(WorkerHeartbeat)
                .where(WorkerHeartbeat.worker_name == identity.worker_name)
                .limit(1)
            ).scalars().first()

            if existing is None:
                heartbeat = WorkerHeartbeat(
                    worker_name=identity.worker_name,
                    host=identity.host,
                    pid=identity.pid,
                    status=status,
                    last_heartbeat_at=now,
                    started_at=now,
                    updated_at=now,
                    metadata_json=dict(metadata_json) if metadata_json is not None else None,
                )
                session.add(heartbeat)
                session.flush()
                row = heartbeat
            else:
                existing.host = identity.host
                existing.pid = identity.pid
                existing.status = status
                existing.last_heartbeat_at = now
                existing.updated_at = now
                if metadata_json is not None:
                    existing.metadata_json = dict(metadata_json)
                row = existing

    return _build_worker_status_record(row, now=now, stale_threshold_seconds=0)


def list_workers(
    *,
    stale_threshold_seconds: int,
    session_factory: sessionmaker[Session] | None = None,
) -> list[WorkerStatusRecord]:
    resolved_session_factory = session_factory or create_session_factory()
    now = utc_now()

    with resolved_session_factory() as session:
        rows = session.execute(
            select(WorkerHeartbeat)
            .order_by(WorkerHeartbeat.last_heartbeat_at.desc(), WorkerHeartbeat.worker_name.asc())
        ).scalars().all()

    return [
        _build_worker_status_record(row, now=now, stale_threshold_seconds=stale_threshold_seconds)
        for row in rows
    ]


def find_stale_runs(
    *,
    stale_threshold_seconds: int,
    max_running_age_seconds: int | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> list[StaleRunCandidate]:
    resolved_session_factory = session_factory or create_session_factory()
    now = utc_now()
    max_running_age = _resolve_max_running_age_seconds(
        stale_threshold_seconds=stale_threshold_seconds,
        max_running_age_seconds=max_running_age_seconds,
    )

    with resolved_session_factory() as session:
        rows = _fetch_running_run_contexts(session)

    stale_candidates: list[StaleRunCandidate] = []
    for row in rows:
        candidate = _evaluate_stale_run_candidate(
            row,
            now=now,
            stale_threshold_seconds=stale_threshold_seconds,
            max_running_age_seconds=max_running_age,
        )
        if candidate is not None:
            stale_candidates.append(candidate)

    stale_candidates.sort(key=lambda item: item.stale_age_seconds, reverse=True)
    return stale_candidates


def recover_stale_runs(
    *,
    stale_threshold_seconds: int,
    max_running_age_seconds: int | None = None,
    recovered_by_worker: str = "recovery-command",
    dry_run: bool = False,
    session_factory: sessionmaker[Session] | None = None,
) -> RecoverStaleRunsResult:
    resolved_session_factory = session_factory or create_session_factory()
    stale_candidates = find_stale_runs(
        stale_threshold_seconds=stale_threshold_seconds,
        max_running_age_seconds=max_running_age_seconds,
        session_factory=resolved_session_factory,
    )

    if dry_run:
        return RecoverStaleRunsResult(
            candidate_count=len(stale_candidates),
            recovered_count=0,
            skipped_count=0,
            stale_runs=tuple(stale_candidates),
        )

    now = utc_now()
    max_running_age = _resolve_max_running_age_seconds(
        stale_threshold_seconds=stale_threshold_seconds,
        max_running_age_seconds=max_running_age_seconds,
    )

    recovered_count = 0
    skipped_count = 0

    with resolved_session_factory() as session:
        with session.begin():
            for candidate in stale_candidates:
                context = _fetch_running_run_context(session, candidate.run_id)
                if context is None:
                    skipped_count += 1
                    continue

                reevaluated = _evaluate_stale_run_candidate(
                    context,
                    now=now,
                    stale_threshold_seconds=stale_threshold_seconds,
                    max_running_age_seconds=max_running_age,
                )
                if reevaluated is None:
                    skipped_count += 1
                    continue

                run = context.run
                previous_status = run.status
                previous_claimed_by_worker = run.claimed_by_worker
                run.status = RUN_STATUS_PENDING
                run.claimed_by_worker = None
                run.claimed_at = None
                run.started_at = None
                run.ended_at = None
                run.updated_at = now

                record_run_events(
                    run.id,
                    events=(
                        RunEventCreate(
                            event_type=RUN_EVENT_RUN_MARKED_STALE,
                            message="Run identified as stale.",
                            payload_json={
                                "worker_name": previous_claimed_by_worker,
                                "reason": reevaluated.reason,
                            },
                        ),
                        RunEventCreate(
                            event_type=RUN_EVENT_RUN_RECOVERED_TO_PENDING,
                            message="Stale run recovered to pending.",
                            payload_json={
                                "worker_name": recovered_by_worker,
                                "previous_status": previous_status,
                            },
                        ),
                    ),
                    session=session,
                )
                recovered_count += 1

    return RecoverStaleRunsResult(
        candidate_count=len(stale_candidates),
        recovered_count=recovered_count,
        skipped_count=skipped_count,
        stale_runs=tuple(stale_candidates),
    )


def mark_run_claimed(
    run: AgentRun,
    *,
    worker_name: str | None,
    claimed_at: datetime | None = None,
) -> None:
    run.claimed_by_worker = worker_name
    run.claimed_at = claimed_at or utc_now()


def clear_run_claim(run: AgentRun) -> None:
    run.claimed_by_worker = None
    run.claimed_at = None


@dataclass(frozen=True)
class _RunningRunContext:
    run: AgentRun
    worker_heartbeat: WorkerHeartbeat | None


def _fetch_running_run_contexts(session: Session) -> list[_RunningRunContext]:
    rows = session.execute(
        select(AgentRun, WorkerHeartbeat)
        .outerjoin(
            WorkerHeartbeat,
            AgentRun.claimed_by_worker == WorkerHeartbeat.worker_name,
        )
        .where(AgentRun.status == RUN_STATUS_RUNNING)
        .order_by(AgentRun.updated_at.asc(), AgentRun.id.asc())
    ).all()
    return [
        _RunningRunContext(run=row[0], worker_heartbeat=row[1])
        for row in rows
    ]


def _fetch_running_run_context(session: Session, run_id: uuid.UUID) -> _RunningRunContext | None:
    row = session.execute(
        select(AgentRun, WorkerHeartbeat)
        .outerjoin(
            WorkerHeartbeat,
            AgentRun.claimed_by_worker == WorkerHeartbeat.worker_name,
        )
        .where(AgentRun.id == run_id)
        .with_for_update()
        .limit(1)
    ).one_or_none()
    if row is None:
        return None

    return _RunningRunContext(run=row[0], worker_heartbeat=row[1])


def _evaluate_stale_run_candidate(
    context: _RunningRunContext,
    *,
    now: datetime,
    stale_threshold_seconds: int,
    max_running_age_seconds: int,
) -> StaleRunCandidate | None:
    run = context.run
    heartbeat = context.worker_heartbeat
    if run.status != RUN_STATUS_RUNNING:
        return None

    run_age_seconds = _age_seconds(now, _run_age_anchor(run))
    if run_age_seconds < stale_threshold_seconds:
        return None

    reason: str | None = None
    max_age_exceeded = run_age_seconds >= max_running_age_seconds

    if run.claimed_by_worker is None:
        reason = "missing claimed worker metadata"
    elif heartbeat is None:
        reason = "claiming worker missing heartbeat row"
    else:
        if heartbeat.status == WORKER_STATUS_STOPPED:
            reason = "claiming worker is stopped"
        heartbeat_age_seconds = _age_seconds(now, heartbeat.last_heartbeat_at)
        if heartbeat_age_seconds >= stale_threshold_seconds:
            reason = "worker heartbeat expired"

    if max_age_exceeded:
        reason = "run exceeded maximum running age"

    if reason is None:
        return None

    return StaleRunCandidate(
        run_id=run.id,
        agent_id=run.agent_id,
        claimed_by_worker=run.claimed_by_worker,
        claimed_at=run.claimed_at,
        started_at=run.started_at,
        updated_at=run.updated_at,
        worker_last_heartbeat_at=heartbeat.last_heartbeat_at if heartbeat is not None else None,
        worker_status=heartbeat.status if heartbeat is not None else None,
        stale_age_seconds=run_age_seconds,
        reason=reason,
    )


def _run_age_anchor(run: AgentRun) -> datetime:
    if run.claimed_at is not None:
        return run.claimed_at
    if run.started_at is not None:
        return run.started_at
    return run.updated_at


def _resolve_max_running_age_seconds(*, stale_threshold_seconds: int, max_running_age_seconds: int | None) -> int:
    if max_running_age_seconds is not None and max_running_age_seconds > 0:
        return max_running_age_seconds

    return stale_threshold_seconds * DEFAULT_MAX_RUNNING_AGE_MULTIPLIER


def _build_worker_status_record(
    row: WorkerHeartbeat,
    *,
    now: datetime,
    stale_threshold_seconds: int,
) -> WorkerStatusRecord:
    heartbeat_age_seconds = _age_seconds(now, row.last_heartbeat_at)
    freshness = "active"
    if row.status == WORKER_STATUS_STOPPED or heartbeat_age_seconds >= stale_threshold_seconds > 0:
        freshness = "stale"

    return WorkerStatusRecord(
        worker_id=row.id,
        worker_name=row.worker_name,
        host=row.host,
        pid=row.pid,
        status=row.status,
        last_heartbeat_at=row.last_heartbeat_at,
        started_at=row.started_at,
        updated_at=row.updated_at,
        metadata_json=dict(row.metadata_json) if row.metadata_json is not None else None,
        heartbeat_age_seconds=heartbeat_age_seconds,
        freshness=freshness,
    )


def _age_seconds(now: datetime, value: datetime) -> int:
    delta = now - value
    return max(int(delta.total_seconds()), 0)
