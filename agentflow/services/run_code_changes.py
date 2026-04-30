from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from agentflow.db.models import AgentRun, RunCodeChange, utc_now
from agentflow.db.session import create_session_factory


@dataclass(frozen=True)
class RunCodeChangeRecord:
    code_change_id: uuid.UUID
    run_id: uuid.UUID
    base_commit_sha: str | None
    result_commit_sha: str | None
    commit_message: str | None
    changed_files_json: list[dict[str, Any]]
    created_at: datetime


def create_run_code_change(
    run_id: uuid.UUID,
    *,
    base_commit_sha: str | None,
    result_commit_sha: str | None,
    commit_message: str | None,
    changed_files_json: list[dict[str, Any]],
    session_factory: sessionmaker[Session] | None = None,
) -> RunCodeChangeRecord:
    session_factory = session_factory or create_session_factory()
    now = utc_now()

    with session_factory() as session:
        with session.begin():
            if session.get(AgentRun, run_id) is None:
                raise ValueError(f"Run not found: {run_id}")

            row = RunCodeChange(
                run_id=run_id,
                base_commit_sha=base_commit_sha,
                result_commit_sha=result_commit_sha,
                commit_message=commit_message,
                changed_files_json=[dict(item) for item in changed_files_json],
                created_at=now,
            )
            session.add(row)
            session.flush()
            return _build_code_change_record(row)


def get_latest_run_code_change(
    run_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> RunCodeChangeRecord | None:
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        row = session.execute(
            select(RunCodeChange)
            .where(RunCodeChange.run_id == run_id)
            .order_by(RunCodeChange.created_at.desc(), RunCodeChange.id.desc())
            .limit(1)
        ).scalars().first()

    if row is None:
        return None

    return _build_code_change_record(row)


def _build_code_change_record(row: RunCodeChange) -> RunCodeChangeRecord:
    return RunCodeChangeRecord(
        code_change_id=row.id,
        run_id=row.run_id,
        base_commit_sha=row.base_commit_sha,
        result_commit_sha=row.result_commit_sha,
        commit_message=row.commit_message,
        changed_files_json=[dict(item) for item in row.changed_files_json],
        created_at=row.created_at,
    )
