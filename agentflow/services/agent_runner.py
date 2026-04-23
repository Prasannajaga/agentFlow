from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from agentflow.services.agent_queries import get_registered_agent
from agentflow.services.fake_provider import execute_fake_agent
from agentflow.services.run_queries import (
    AgentRunDetail,
    create_agent_run,
    get_agent_execution_target,
    mark_agent_run_completed,
    mark_agent_run_failed,
    mark_agent_run_running,
)

RunExecutor = Callable[[dict[str, Any]], dict[str, Any]]


class AgentRunError(RuntimeError):
    """Base error for synchronous agent execution failures."""


class AgentNotFoundError(AgentRunError):
    def __init__(self, agent_id: uuid.UUID):
        super().__init__(f"Agent not found: {agent_id}")
        self.agent_id = agent_id


class AgentVersionNotFoundError(AgentRunError):
    def __init__(self, agent_id: uuid.UUID):
        super().__init__(f"No versions found for agent: {agent_id}")
        self.agent_id = agent_id


class AgentRunExecutionFailedError(AgentRunError):
    def __init__(self, run: AgentRunDetail):
        message = run.error_message or "Synchronous agent execution failed."
        super().__init__(message)
        self.run = run


def run_registered_agent(
    agent_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
    executor: RunExecutor | None = None,
) -> AgentRunDetail:
    agent = get_registered_agent(agent_id, session_factory=session_factory)
    if agent is None:
        raise AgentNotFoundError(agent_id)

    if agent.latest_version is None:
        raise AgentVersionNotFoundError(agent_id)

    execution_target = get_agent_execution_target(agent_id, session_factory=session_factory)
    if execution_target is None:
        raise AgentVersionNotFoundError(agent_id)

    run = create_agent_run(
        agent_id=execution_target.agent_id,
        version_id=execution_target.version_id,
        resolved_config_json=execution_target.normalized_config_json,
        session_factory=session_factory,
    )

    resolved_executor = executor or execute_fake_agent

    try:
        running_run = mark_agent_run_running(run.run_id, session_factory=session_factory)
        if running_run is None:
            raise RuntimeError(f"Run not found after creation: {run.run_id}")

        output_json = resolved_executor(execution_target.normalized_config_json)
        completed_run = mark_agent_run_completed(
            run.run_id,
            output_json=output_json,
            session_factory=session_factory,
        )
        if completed_run is None:
            raise RuntimeError(f"Run not found while finalizing completion: {run.run_id}")

        return completed_run
    except Exception as exc:
        failed_run = mark_agent_run_failed(
            run.run_id,
            error_message=str(exc),
            session_factory=session_factory,
        )
        if failed_run is None:
            raise

        raise AgentRunExecutionFailedError(failed_run) from exc
