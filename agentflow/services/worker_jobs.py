from __future__ import annotations

import os
import socket
import sys
import time

from sqlalchemy import literal, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from agentflow.config import get_worker_heartbeat_interval_seconds
from agentflow.db.session import create_session_factory
from agentflow.services.agent_runner import (
    AgentRunExecutionFailedError,
    AgentRunNotFoundError,
    execute_claimed_run,
)
from agentflow.services.run_queries import claim_next_pending_run
from agentflow.services.worker_ops import (
    WORKER_STATUS_IDLE,
    WORKER_STATUS_PROCESSING,
    WORKER_STATUS_STARTING,
    WORKER_STATUS_STOPPED,
    WorkerIdentity,
    heartbeat_worker,
)

DEFAULT_WORKER_POLL_INTERVAL_SECONDS = 1.0


def start_worker_loop(
    *,
    poll_interval_seconds: float = DEFAULT_WORKER_POLL_INTERVAL_SECONDS,
    session_factory: sessionmaker[Session] | None = None,
) -> None:
    resolved_session_factory = session_factory or create_session_factory()
    worker_identity = build_worker_identity()
    heartbeat_interval_seconds = float(get_worker_heartbeat_interval_seconds())
    ensure_worker_database_ready(resolved_session_factory)

    _safe_heartbeat(
        worker_identity,
        status=WORKER_STATUS_STARTING,
        session_factory=resolved_session_factory,
    )
    _safe_heartbeat(
        worker_identity,
        status=WORKER_STATUS_IDLE,
        session_factory=resolved_session_factory,
    )

    print(f"Worker started. Polling Postgres for pending runs every {poll_interval_seconds:.1f}s.")
    print(
        f"Worker identity: {worker_identity.worker_name} "
        f"(heartbeat every {heartbeat_interval_seconds:.1f}s)"
    )

    next_heartbeat_at = 0.0
    try:
        while True:
            monotonic_now = time.monotonic()
            if monotonic_now >= next_heartbeat_at:
                _safe_heartbeat(
                    worker_identity,
                    status=WORKER_STATUS_IDLE,
                    session_factory=resolved_session_factory,
                )
                next_heartbeat_at = monotonic_now + heartbeat_interval_seconds

            try:
                claimed_run = claim_next_pending_run(
                    worker_id=worker_identity.worker_name,
                    session_factory=resolved_session_factory,
                )
            except SQLAlchemyError as exc:
                print("Worker polling failed due to a database error.", file=sys.stderr)
                print(f"- {format_database_error_message(exc)}", file=sys.stderr)
                time.sleep(poll_interval_seconds)
                continue

            if claimed_run is None:
                time.sleep(poll_interval_seconds)
                continue

            print(
                f"Picked run {claimed_run.run_id} "
                f"attempt={claimed_run.attempt_count}/{claimed_run.max_attempts}"
            )
            _safe_heartbeat(
                worker_identity,
                status=WORKER_STATUS_PROCESSING,
                metadata_json={"run_id": str(claimed_run.run_id)},
                session_factory=resolved_session_factory,
            )

            try:
                completed_run = execute_claimed_run(
                    claimed_run,
                    session_factory=resolved_session_factory,
                )
            except AgentRunExecutionFailedError as exc:
                print(
                    f"run_id={exc.run.run_id} status={exc.run.status} error={exc.run.error_message}",
                    file=sys.stderr,
                )
                continue
            except AgentRunNotFoundError as exc:
                print(str(exc), file=sys.stderr)
                continue
            except Exception as exc:  # pragma: no cover - defensive worker guard
                print(f"run_id={claimed_run.run_id} unexpected worker error: {exc}", file=sys.stderr)
                continue
            finally:
                _safe_heartbeat(
                    worker_identity,
                    status=WORKER_STATUS_IDLE,
                    metadata_json={"last_run_id": str(claimed_run.run_id)},
                    session_factory=resolved_session_factory,
                )

            print(
                f"run_id={completed_run.run_id} status={completed_run.status} "
                f"attempt={completed_run.attempt_count}/{completed_run.max_attempts}"
            )
    finally:
        _safe_heartbeat(
            worker_identity,
            status=WORKER_STATUS_STOPPED,
            session_factory=resolved_session_factory,
        )


def ensure_worker_database_ready(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        session.execute(select(literal(1))).scalar_one()


def build_worker_identity() -> WorkerIdentity:
    host = socket.gethostname()
    pid = os.getpid()
    return WorkerIdentity(
        worker_name=f"{host}:{pid}",
        host=host,
        pid=pid,
    )


def format_database_error_message(exc: SQLAlchemyError) -> str:
    original = getattr(exc, "orig", None)
    if original is not None:
        return str(original)

    return str(exc)


def _safe_heartbeat(
    identity: WorkerIdentity,
    *,
    status: str,
    metadata_json: dict[str, str] | None = None,
    session_factory: sessionmaker[Session],
) -> None:
    try:
        heartbeat_worker(
            identity,
            status=status,
            metadata_json=metadata_json,
            session_factory=session_factory,
        )
    except SQLAlchemyError as exc:  # pragma: no cover - operational warning path
        print("Worker heartbeat update failed due to a database error.", file=sys.stderr)
        print(f"- {format_database_error_message(exc)}", file=sys.stderr)
