from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from agentflow.db.models import AgentRun, utc_now
from agentflow.db.session import create_session_factory
from agentflow.services.retry_policy import should_retry_failed_attempt
from agentflow.services.run_events import (
    RUN_EVENT_RUN_MANUAL_RETRY_REQUESTED,
    RUN_EVENT_RUN_RETRY_SCHEDULED,
    RunEventCreate,
    record_run_events,
)
from agentflow.services.run_queries import (
    RUN_STATUS_FAILED,
    RUN_STATUS_PENDING,
    AgentRunDetail,
    get_agent_run,
)


@dataclass(frozen=True)
class RetryEligibility:
    eligible: bool
    reason: str | None = None


class RunActionError(RuntimeError):
    """Base error for viewer/CLI run actions."""


class RunActionNotFoundError(RunActionError):
    def __init__(self, run_id: uuid.UUID):
        super().__init__(f"Run not found: {run_id}")
        self.run_id = run_id


class RunRetryNotEligibleError(RunActionError):
    def __init__(self, run_id: uuid.UUID, reason: str):
        super().__init__(reason)
        self.run_id = run_id
        self.reason = reason


def get_manual_retry_eligibility(run: AgentRunDetail) -> RetryEligibility:
    if run.status != RUN_STATUS_FAILED:
        return RetryEligibility(False, "Only failed runs can be retried manually.")

    if not run.retryable:
        return RetryEligibility(False, "This run is marked non-retryable.")

    if not should_retry_failed_attempt(
        attempt_count=run.attempt_count,
        max_attempts=run.max_attempts,
        retryable=run.retryable,
    ):
        return RetryEligibility(False, "No retry attempts remain.")

    return RetryEligibility(True)


def manual_retry_run(
    run_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> AgentRunDetail:
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        with session.begin():
            run = session.get(AgentRun, run_id)
            if run is None:
                raise RunActionNotFoundError(run_id)

            eligibility = get_manual_retry_eligibility(_build_action_run_detail(run))
            if not eligibility.eligible:
                raise RunRetryNotEligibleError(run_id, eligibility.reason or "Run is not eligible for retry.")

            now = utc_now()
            run.status = RUN_STATUS_PENDING
            run.ended_at = None
            run.updated_at = now
            record_run_events(
                run.id,
                events=(
                    RunEventCreate(
                        event_type=RUN_EVENT_RUN_MANUAL_RETRY_REQUESTED,
                        message="Manual retry requested.",
                        payload_json={
                            "attempt_count": run.attempt_count,
                            "max_attempts": run.max_attempts,
                        },
                    ),
                    RunEventCreate(
                        event_type=RUN_EVENT_RUN_RETRY_SCHEDULED,
                        message="Run queued for manual retry.",
                        payload_json={
                            "attempt_count": run.attempt_count,
                            "max_attempts": run.max_attempts,
                            "next_status": RUN_STATUS_PENDING,
                        },
                    ),
                ),
                session=session,
            )

        refreshed = get_agent_run(run_id, session_factory=session_factory)

    if refreshed is None:
        raise RunActionNotFoundError(run_id)
    return refreshed


def _build_action_run_detail(run: AgentRun) -> AgentRunDetail:
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
        created_at=run.created_at,
        started_at=run.started_at,
        ended_at=run.ended_at,
        updated_at=run.updated_at,
    )
