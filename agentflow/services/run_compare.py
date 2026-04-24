from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from agentflow.services.run_events import get_run_event_summary
from agentflow.services.run_queries import AgentRunDetail, get_agent_run


class RunCompareError(RuntimeError):
    """Base error for run comparison."""


class RunCompareInvalidError(RunCompareError):
    pass


class RunCompareNotFoundError(RunCompareError):
    pass


@dataclass(frozen=True)
class ComparableRun:
    run: AgentRunDetail
    event_count: int


@dataclass(frozen=True)
class RunComparison:
    agent_id: uuid.UUID
    runs: list[ComparableRun]


def compare_runs(
    run_ids: list[uuid.UUID],
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> RunComparison:
    if len(run_ids) < 2:
        raise RunCompareInvalidError("At least two run IDs are required for comparison.")

    runs: list[ComparableRun] = []
    for run_id in run_ids:
        run = get_agent_run(run_id, session_factory=session_factory)
        if run is None:
            raise RunCompareNotFoundError(f"Run not found: {run_id}")
        event_summary = get_run_event_summary(run_id, session_factory=session_factory)
        runs.append(ComparableRun(run=run, event_count=event_summary.event_count))

    agent_ids = {item.run.agent_id for item in runs}
    if len(agent_ids) != 1:
        raise RunCompareInvalidError("Runs must belong to the same agent.")

    return RunComparison(agent_id=runs[0].run.agent_id, runs=runs)
