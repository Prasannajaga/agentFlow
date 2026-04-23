from __future__ import annotations

import os
import socket
import sys
import time

from sqlalchemy import literal, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from agentflow.db.session import create_session_factory
from agentflow.services.agent_runner import (
    AgentRunExecutionFailedError,
    AgentRunNotFoundError,
    execute_claimed_run,
)
from agentflow.services.run_queries import claim_next_pending_run

DEFAULT_WORKER_POLL_INTERVAL_SECONDS = 1.0


def start_worker_loop(
    *,
    poll_interval_seconds: float = DEFAULT_WORKER_POLL_INTERVAL_SECONDS,
    session_factory: sessionmaker[Session] | None = None,
) -> None:
    resolved_session_factory = session_factory or create_session_factory()
    worker_id = build_worker_id()
    ensure_worker_database_ready(resolved_session_factory)

    print(f"Worker started. Polling Postgres for pending runs every {poll_interval_seconds:.1f}s.")

    while True:
        try:
            claimed_run = claim_next_pending_run(
                worker_id=worker_id,
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

        print(f"Picked run {claimed_run.run_id}")

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

        print(f"run_id={completed_run.run_id} status={completed_run.status}")


def ensure_worker_database_ready(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        session.execute(select(literal(1))).scalar_one()


def build_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def format_database_error_message(exc: SQLAlchemyError) -> str:
    original = getattr(exc, "orig", None)
    if original is not None:
        return str(original)

    return str(exc)
