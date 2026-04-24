from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from agentflow.providers.base import (
    ProviderError,
    ProviderExecutionError,
    ProviderInvocationRequest,
    ProviderResult,
    preview_text,
)
from agentflow.providers.registry import get_provider_adapter
from agentflow.services.fake_provider import execute_fake_agent
from agentflow.services.agent_queries import get_registered_agent
from agentflow.services.run_queries import (
    RUN_STATUS_PENDING,
    RUN_STATUS_RUNNING,
    TERMINAL_RUN_STATUSES,
    AgentRunDetail,
    create_agent_run,
    get_agent_execution_target,
    get_agent_run,
    mark_agent_run_completed,
    mark_agent_run_failed,
    mark_agent_run_running,
)
from agentflow.services.run_events import (
    RUN_EVENT_TOOL_EXECUTION_COMPLETED,
    RUN_EVENT_TOOL_EXECUTION_FAILED,
    RUN_EVENT_TOOL_EXECUTION_STARTED,
    RUN_EVENT_TOOLS_VALIDATED,
    RUN_EVENT_PROVIDER_EXECUTION_FAILED,
    RUN_EVENT_PROVIDER_EXECUTION_COMPLETED,
    RUN_EVENT_PROVIDER_REQUEST_PREPARED,
    RUN_EVENT_PROVIDER_EXECUTION_STARTED,
    RUN_EVENT_RUN_COMPLETED,
    RUN_EVENT_RUN_FAILED,
    RUN_EVENT_RUN_STARTED,
    RunEventCreate,
    record_run_event,
)
from agentflow.tools.base import ToolError, ToolExecutionError, ToolInvocationRequest, ToolResult
from agentflow.tools.echo import ECHO_TOOL_NAME, build_echo_input, preview_echo_result
from agentflow.tools.registry import get_tool_adapter, validate_tool_names

RunExecutor = Callable[[dict[str, Any]], dict[str, Any]]


class AgentRunError(RuntimeError):
    """Base error for agent execution lifecycle failures."""


class AgentNotFoundError(AgentRunError):
    def __init__(self, agent_id: uuid.UUID):
        super().__init__(f"Agent not found: {agent_id}")
        self.agent_id = agent_id


class AgentVersionNotFoundError(AgentRunError):
    def __init__(self, agent_id: uuid.UUID):
        super().__init__(f"No versions found for agent: {agent_id}")
        self.agent_id = agent_id


class AgentRunNotFoundError(AgentRunError):
    def __init__(self, run_id: uuid.UUID):
        super().__init__(f"Run not found: {run_id}")
        self.run_id = run_id


class AgentRunExecutionFailedError(AgentRunError):
    def __init__(self, run: AgentRunDetail):
        message = run.error_message or "Agent execution failed."
        super().__init__(message)
        self.run = run


@dataclass(frozen=True)
class PreparedAgentRun:
    run: AgentRunDetail


def create_run_for_agent(
    agent_id: uuid.UUID,
    *,
    input_json: dict[str, Any] | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> PreparedAgentRun:
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
        input_json=input_json,
        session_factory=session_factory,
    )

    return PreparedAgentRun(run=run)


def execute_agent_run(
    run_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
    executor: RunExecutor | None = None,
) -> AgentRunDetail:
    run = get_agent_run(run_id, session_factory=session_factory)
    if run is None:
        raise AgentRunNotFoundError(run_id)

    if run.status in TERMINAL_RUN_STATUSES or run.status == RUN_STATUS_RUNNING:
        if run.status in TERMINAL_RUN_STATUSES:
            return run
        return execute_claimed_run(run, session_factory=session_factory, executor=executor)

    if run.status == RUN_STATUS_PENDING:
        running_run = mark_agent_run_running(
            run.run_id,
            events=(
                RunEventCreate(
                    event_type=RUN_EVENT_RUN_STARTED,
                    message="Run moved to running state.",
                    payload_json={"status": RUN_STATUS_RUNNING},
                ),
            ),
            session_factory=session_factory,
        )
        if running_run is None:
            raise AgentRunNotFoundError(run.run_id)

        return execute_claimed_run(running_run, session_factory=session_factory, executor=executor)

    raise AgentRunError(f"Run {run.run_id} has unsupported status: {run.status}")


def execute_claimed_run(
    run: AgentRunDetail,
    *,
    session_factory: sessionmaker[Session] | None = None,
    executor: RunExecutor | None = None,
) -> AgentRunDetail:
    if run.status in TERMINAL_RUN_STATUSES:
        return run

    if run.status != RUN_STATUS_RUNNING:
        raise AgentRunError(f"Run {run.run_id} must be running before execution.")

    request: ProviderInvocationRequest | None = None
    provider_type = _peek_provider_type(run.resolved_config_json)
    tool_results: list[ToolResult] = []

    try:
        tool_results = _execute_tools_for_run(run, session_factory=session_factory)
        request = ProviderInvocationRequest.from_resolved_config(
            run.resolved_config_json,
            input_json=run.input_json,
        )
        provider_type = request.provider_type
        record_run_event(
            run.run_id,
            event_type=RUN_EVENT_PROVIDER_EXECUTION_STARTED,
            message="Provider execution started.",
            payload_json={
                "provider_type": provider_type,
                "model": request.model,
            },
            session_factory=session_factory,
        )

        prepared_payload = _build_provider_request_payload(request, executor=executor)
        record_run_event(
            run.run_id,
            event_type=RUN_EVENT_PROVIDER_REQUEST_PREPARED,
            message="Provider request prepared.",
            payload_json=prepared_payload,
            session_factory=session_factory,
        )
        provider_result = _invoke_provider_for_run(run, request=request, executor=executor)
        output_json = provider_result.to_output_json()
        if tool_results:
            output_json["tool_results"] = [tool_result.to_json() for tool_result in tool_results]
        completed_run = mark_agent_run_completed(
            run.run_id,
            output_json=output_json,
            events=(
                RunEventCreate(
                    event_type=RUN_EVENT_PROVIDER_EXECUTION_COMPLETED,
                    message="Provider execution completed.",
                    payload_json={
                        "provider_type": provider_result.provider_type,
                        "model": provider_result.model,
                        "output_preview": preview_text(provider_result.output_text),
                    },
                ),
                RunEventCreate(
                    event_type=RUN_EVENT_RUN_COMPLETED,
                    message="Run completed successfully.",
                    payload_json={
                        "status": "completed",
                        "provider_type": provider_result.provider_type,
                        "model": provider_result.model,
                    },
                ),
            ),
            session_factory=session_factory,
        )
        if completed_run is None:
            raise AgentRunNotFoundError(run.run_id)

        return completed_run
    except ToolError as exc:
        failed_run = mark_agent_run_failed(
            run.run_id,
            error_message=str(exc),
            events=(
                RunEventCreate(
                    event_type=RUN_EVENT_TOOL_EXECUTION_FAILED,
                    message="Tool execution failed.",
                    payload_json={
                        "tool_name": exc.tool_name,
                        "error_type": exc.error_type,
                        "message": str(exc),
                    },
                ),
                RunEventCreate(
                    event_type=RUN_EVENT_RUN_FAILED,
                    message="Run failed during execution.",
                    payload_json={
                        "tool_name": exc.tool_name,
                        "error_message": str(exc),
                    },
                ),
            ),
            session_factory=session_factory,
        )
        if failed_run is None:
            raise

        raise AgentRunExecutionFailedError(failed_run) from exc
    except ProviderError as exc:
        failed_run = mark_agent_run_failed(
            run.run_id,
            error_message=str(exc),
            events=(
                RunEventCreate(
                    event_type=RUN_EVENT_PROVIDER_EXECUTION_FAILED,
                    message="Provider execution failed.",
                    payload_json={
                        "provider_type": exc.provider_type,
                        "error_type": exc.error_type,
                        "message": str(exc),
                    },
                ),
                RunEventCreate(
                    event_type=RUN_EVENT_RUN_FAILED,
                    message="Run failed during execution.",
                    payload_json={
                        "provider_type": exc.provider_type,
                        "error_message": str(exc),
                    },
                ),
            ),
            session_factory=session_factory,
        )
        if failed_run is None:
            raise

        raise AgentRunExecutionFailedError(failed_run) from exc
    except Exception as exc:
        provider_error = ProviderExecutionError(
            provider_type,
            f"Unexpected provider execution error: {exc}",
            error_type="unexpected_error",
        )
        failed_run = mark_agent_run_failed(
            run.run_id,
            error_message=str(provider_error),
            events=(
                RunEventCreate(
                    event_type=RUN_EVENT_PROVIDER_EXECUTION_FAILED,
                    message="Provider execution failed.",
                    payload_json={
                        "provider_type": provider_error.provider_type,
                        "error_type": provider_error.error_type,
                        "message": str(provider_error),
                    },
                ),
                RunEventCreate(
                    event_type=RUN_EVENT_RUN_FAILED,
                    message="Run failed during execution.",
                    payload_json={
                        "provider_type": provider_error.provider_type,
                        "error_message": str(provider_error),
                    },
                ),
            ),
            session_factory=session_factory,
        )
        if failed_run is None:
            raise

        raise AgentRunExecutionFailedError(failed_run) from exc


def _invoke_provider_for_run(
    run: AgentRunDetail,
    *,
    request: ProviderInvocationRequest,
    executor: RunExecutor | None,
) -> ProviderResult:
    if executor is not None:
        output_json = executor(run.resolved_config_json)
        return _normalize_legacy_provider_output(output_json, request=request)

    adapter = get_provider_adapter(request)
    return adapter.invoke(request)


def _build_provider_request_payload(
    request: ProviderInvocationRequest,
    *,
    executor: RunExecutor | None,
) -> dict[str, Any]:
    if executor is not None:
        payload: dict[str, Any] = {
            "provider_type": request.provider_type,
            "model": request.model,
        }
        if request.base_url is not None:
            payload["base_url"] = request.base_url
        return payload

    adapter = get_provider_adapter(request)
    return adapter.describe_request(request)


def _normalize_legacy_provider_output(
    output_json: dict[str, Any],
    *,
    request: ProviderInvocationRequest,
) -> ProviderResult:
    provider_type = str(output_json.get("provider_type") or output_json.get("provider") or request.provider_type)
    model = str(output_json.get("model") or request.model or "")
    output_text = str(
        output_json.get("output_text")
        or output_json.get("message")
        or "Execution completed."
    )
    finish_reason = output_json.get("finish_reason")

    raw_response = output_json.get("raw_response")
    if not isinstance(raw_response, dict):
        raw_response = {
            key: value
            for key, value in output_json.items()
            if key not in {"provider_type", "provider", "model", "output_text", "message", "finish_reason"}
        } or None

    return ProviderResult(
        provider_type=provider_type,
        model=model,
        output_text=output_text,
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        raw_response=raw_response,
    )


def _peek_provider_type(resolved_config_json: dict[str, Any]) -> str:
    provider_config = resolved_config_json.get("provider")
    if isinstance(provider_config, dict):
        provider_type = provider_config.get("type")
        if isinstance(provider_type, str) and provider_type.strip():
            return provider_type.strip()
    return "unknown"


def _execute_tools_for_run(
    run: AgentRunDetail,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> list[ToolResult]:
    configured_tools = _extract_configured_tools(run.resolved_config_json)
    if not configured_tools:
        return []

    validated_tools = validate_tool_names(configured_tools)
    record_run_event(
        run.run_id,
        event_type=RUN_EVENT_TOOLS_VALIDATED,
        message="Configured tools validated.",
        payload_json={"tools": list(validated_tools)},
        session_factory=session_factory,
    )

    tool_results: list[ToolResult] = []
    if ECHO_TOOL_NAME in validated_tools:
        tool_results.append(_execute_echo_tool(run, session_factory=session_factory))

    return tool_results


def _execute_echo_tool(
    run: AgentRunDetail,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> ToolResult:
    tool_name = ECHO_TOOL_NAME
    record_run_event(
        run.run_id,
        event_type=RUN_EVENT_TOOL_EXECUTION_STARTED,
        message="Tool execution started.",
        payload_json={"tool_name": tool_name},
        session_factory=session_factory,
    )

    try:
        tool_adapter = get_tool_adapter(tool_name)
        tool_request = ToolInvocationRequest(
            tool_name=tool_name,
            input_data=build_echo_input(run.input_json),
        )
        tool_result = tool_adapter.invoke(tool_request)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolExecutionError(
            tool_name,
            f"Unexpected tool execution error: {exc}",
            error_type="unexpected_error",
        ) from exc

    record_run_event(
        run.run_id,
        event_type=RUN_EVENT_TOOL_EXECUTION_COMPLETED,
        message="Tool execution completed.",
        payload_json={
            "tool_name": tool_result.tool_name,
            "ok": tool_result.ok,
            "result_preview": preview_echo_result(tool_result),
        },
        session_factory=session_factory,
    )
    return tool_result


def _extract_configured_tools(resolved_config_json: dict[str, Any]) -> list[str]:
    tools = resolved_config_json.get("tools")
    if not isinstance(tools, list):
        return []
    return [tool_name for tool_name in tools if isinstance(tool_name, str)]
