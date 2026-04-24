from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from agentflow.db.models import AgentDefinition, AgentRun
from agentflow.db.session import create_session_factory
from agentflow.services.run_queries import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_PENDING,
    RUN_STATUS_RUNNING,
)


@dataclass(frozen=True)
class DashboardStats:
    agents_total: int
    runs_total: int
    runs_pending: int
    runs_running: int
    runs_completed: int
    runs_failed: int


def get_dashboard_stats(
    session_factory: sessionmaker[Session] | None = None,
) -> DashboardStats:
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        agents_total = session.execute(select(func.count()).select_from(AgentDefinition)).scalar_one()
        runs_total = session.execute(select(func.count()).select_from(AgentRun)).scalar_one()
        status_rows = session.execute(
            select(AgentRun.status, func.count())
            .group_by(AgentRun.status)
        ).all()

    counts_by_status = {str(status): int(count) for status, count in status_rows}
    return DashboardStats(
        agents_total=int(agents_total),
        runs_total=int(runs_total),
        runs_pending=counts_by_status.get(RUN_STATUS_PENDING, 0),
        runs_running=counts_by_status.get(RUN_STATUS_RUNNING, 0),
        runs_completed=counts_by_status.get(RUN_STATUS_COMPLETED, 0),
        runs_failed=counts_by_status.get(RUN_STATUS_FAILED, 0),
    )
