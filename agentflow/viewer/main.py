from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.exc import SQLAlchemyError

from agentflow.config import ConfigurationError
from agentflow.services.agent_runner import AgentRunNotFoundError, AgentRunSnapshotMissingError, rerun_agent_run
from agentflow.services.agent_queries import (
    AgentDetail,
    AgentSummary,
    AgentVersionSummary,
    get_registered_agent,
    list_agent_versions,
    list_registered_agents,
)
from agentflow.services.run_events import RunEventRecord, list_run_events
from agentflow.services.run_actions import (
    RunActionNotFoundError,
    RunRetryNotEligibleError,
    get_manual_retry_eligibility,
    manual_retry_run,
)
from agentflow.services.run_queries import AgentRunDetail, AgentRunSummary, get_agent_run, list_agent_runs
from agentflow.services.stats_queries import DashboardStats, get_dashboard_stats

VIEWER_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_LIMIT = 50
MAX_RUN_LIMIT = 200

app = FastAPI(title="AgentFlow Local Viewer", version="0.1.0")
app.mount("/static", StaticFiles(directory=VIEWER_DIR / "static"), name="static")
templates = Jinja2Templates(directory=VIEWER_DIR / "templates")


@app.exception_handler(SQLAlchemyError)
async def handle_database_error(request: Request, _exc: SQLAlchemyError) -> HTMLResponse:
    return _error_page(request, status_code=500, title="Database Error", message="The viewer could not read from the database.")


@app.exception_handler(ConfigurationError)
async def handle_configuration_error(request: Request, _exc: ConfigurationError) -> HTMLResponse:
    return _error_page(request, status_code=500, title="Configuration Error", message="Database configuration is incomplete.")


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def dashboard_home(request: Request) -> HTMLResponse:
    stats = get_dashboard_stats()
    return templates.TemplateResponse(
        request,
        "index.html",
        {"page_title": "Dashboard", "stats": stats_to_view(stats)},
    )


@app.get("/agents", response_class=HTMLResponse)
def agents_index(request: Request) -> HTMLResponse:
    agents = [_agent_summary_to_view(agent) for agent in list_registered_agents()]
    return templates.TemplateResponse(
        request,
        "agents.html",
        {"page_title": "Agents", "agents": agents},
    )


@app.get("/agents/{agent_id}", response_class=HTMLResponse)
def agent_detail(request: Request, agent_id: str) -> HTMLResponse:
    parsed_agent_id = _parse_uuid(agent_id)
    if parsed_agent_id is None:
        return _not_found(request, "Agent not found")

    agent = get_registered_agent(parsed_agent_id)
    if agent is None:
        return _not_found(request, "Agent not found")

    latest_version_id = agent.latest_version.version_id if agent.latest_version is not None else None
    versions = [
        {
            **_version_summary_to_view(version),
            "is_latest": version.version_id == latest_version_id,
        }
        for version in (list_agent_versions(parsed_agent_id) or [])
    ]
    return templates.TemplateResponse(
        request,
        "agent_detail.html",
        {
            "page_title": "Agent Detail",
            "agent": _agent_detail_to_view(agent),
            "versions": versions,
        },
    )


@app.get("/runs", response_class=HTMLResponse)
def runs_index(
    request: Request,
    limit: int = Query(DEFAULT_RUN_LIMIT, ge=1, le=MAX_RUN_LIMIT),
) -> HTMLResponse:
    runs = [_run_summary_to_view(run) for run in list_agent_runs(limit=limit)]
    return templates.TemplateResponse(
        request,
        "runs.html",
        {"page_title": "Runs", "runs": runs, "limit": limit},
    )


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: str) -> HTMLResponse:
    parsed_run_id = _parse_uuid(run_id)
    if parsed_run_id is None:
        return _not_found(request, "Run not found")

    run = get_agent_run(parsed_run_id)
    if run is None:
        return _not_found(request, "Run not found")

    events = list_run_events(parsed_run_id)
    retry_eligibility = get_manual_retry_eligibility(run)
    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {
            "page_title": "Run Detail",
            "run": _run_detail_to_view(run),
            "events": [_run_event_to_view(event) for event in events],
            "feedback": _feedback_from_request(request),
            "can_rerun": bool(run.resolved_config_json),
            "retry_eligible": retry_eligibility.eligible,
            "retry_ineligible_reason": retry_eligibility.reason,
        },
    )


@app.post("/runs/{run_id}/rerun")
def rerun_action(request: Request, run_id: str) -> Any:
    parsed_run_id = _parse_uuid(run_id)
    if parsed_run_id is None:
        return _not_found(request, "Run not found")

    try:
        prepared_run = rerun_agent_run(parsed_run_id)
    except AgentRunNotFoundError:
        return _not_found(request, "Run not found")
    except AgentRunSnapshotMissingError as exc:
        return _redirect_to_run(parsed_run_id, error=str(exc))

    return _redirect_to_run(prepared_run.run.run_id, message="Rerun created successfully")


@app.post("/runs/{run_id}/retry")
def retry_action(request: Request, run_id: str) -> Any:
    parsed_run_id = _parse_uuid(run_id)
    if parsed_run_id is None:
        return _not_found(request, "Run not found")

    try:
        retried_run = manual_retry_run(parsed_run_id)
    except RunActionNotFoundError:
        return _not_found(request, "Run not found")
    except RunRetryNotEligibleError as exc:
        return _redirect_to_run(parsed_run_id, error=exc.reason)

    return _redirect_to_run(retried_run.run_id, message="Run queued for retry")


def stats_to_view(stats: DashboardStats) -> dict[str, int]:
    return {
        "agents_total": stats.agents_total,
        "runs_total": stats.runs_total,
        "runs_pending": stats.runs_pending,
        "runs_running": stats.runs_running,
        "runs_completed": stats.runs_completed,
        "runs_failed": stats.runs_failed,
    }


def _agent_summary_to_view(agent: AgentSummary) -> dict[str, Any]:
    return {
        "agent_id": str(agent.agent_id),
        "name": agent.name,
        "description": agent.description,
        "created_at": _format_datetime(agent.created_at),
        "latest_version": _version_summary_to_view(agent.latest_version),
    }


def _agent_detail_to_view(agent: AgentDetail) -> dict[str, Any]:
    return {
        "agent_id": str(agent.agent_id),
        "name": agent.name,
        "description": agent.description,
        "created_at": _format_datetime(agent.created_at),
        "updated_at": _format_datetime(agent.updated_at),
        "latest_version": _version_summary_to_view(agent.latest_version),
    }


def _version_summary_to_view(version: AgentVersionSummary | None) -> dict[str, Any] | None:
    if version is None:
        return None

    return {
        "version_id": str(version.version_id),
        "version_number": version.version_number,
        "config_hash": version.config_hash,
        "created_at": _format_datetime(version.created_at),
    }


def _run_summary_to_view(run: AgentRunSummary) -> dict[str, Any]:
    return {
        "run_id": str(run.run_id),
        "agent_id": str(run.agent_id),
        "version_id": str(run.version_id),
        "source_run_id": str(run.source_run_id) if run.source_run_id is not None else None,
        "status": run.status,
        "attempt_count": run.attempt_count,
        "max_attempts": run.max_attempts,
        "retryable": run.retryable,
        "created_at": _format_datetime(run.created_at),
        "started_at": _format_datetime(run.started_at),
        "ended_at": _format_datetime(run.ended_at),
    }


def _run_detail_to_view(run: AgentRunDetail) -> dict[str, Any]:
    return {
        **_run_summary_to_view(
            AgentRunSummary(
                run_id=run.run_id,
                agent_id=run.agent_id,
                version_id=run.version_id,
                source_run_id=run.source_run_id,
                status=run.status,
                attempt_count=run.attempt_count,
                max_attempts=run.max_attempts,
                retryable=run.retryable,
                created_at=run.created_at,
                started_at=run.started_at,
                ended_at=run.ended_at,
            )
        ),
        "input_json": _format_json(run.input_json),
        "resolved_config_json": _format_json(run.resolved_config_json),
        "output_json": _format_json(run.output_json),
        "output_summary": _summarize_run_output(run.output_json),
        "error_message": run.error_message,
        "last_error_type": run.last_error_type,
        "updated_at": _format_datetime(run.updated_at),
    }


def _run_event_to_view(event: RunEventRecord) -> dict[str, Any]:
    return {
        "id": str(event.event_id),
        "event_type": event.event_type,
        "message": event.message,
        "payload_json": _format_json(event.payload_json),
        "created_at": _format_datetime(event.created_at),
    }


def _summarize_run_output(output_json: dict[str, Any] | None) -> str | None:
    if output_json is None:
        return None

    parts: list[str] = []
    for key in ("provider_type", "provider", "model", "output_text", "message"):
        value = output_json.get(key)
        if value:
            parts.append(str(value))

    return " | ".join(parts[:3]) if parts else "stored"


def _format_json(value: Any) -> str:
    if value is None:
        return "-"
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _format_datetime(value: datetime | None) -> str:
    if value is None:
        return "-"
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


def _feedback_from_request(request: Request) -> dict[str, str | None]:
    return {
        "message": request.query_params.get("message"),
        "error": request.query_params.get("error"),
    }


def _redirect_to_run(run_id: uuid.UUID, *, message: str | None = None, error: str | None = None) -> RedirectResponse:
    target = f"/runs/{run_id}"
    query_parts: list[str] = []
    if message:
        query_parts.append(f"message={quote(message, safe='')}")
    if error:
        query_parts.append(f"error={quote(error, safe='')}")
    if query_parts:
        target = f"{target}?{'&'.join(query_parts)}"
    return RedirectResponse(target, status_code=303)


def _not_found(request: Request, message: str) -> HTMLResponse:
    return _error_page(request, status_code=404, title="Not Found", message=message)


def _error_page(request: Request, *, status_code: int, title: str, message: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "error.html",
        {"page_title": title, "title": title, "message": message, "status_code": status_code},
        status_code=status_code,
    )
