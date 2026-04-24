from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from agentflow.config import ConfigurationError, get_worker_stale_threshold_seconds
from agentflow.services.agent_registry import (
    AgentRegistrationAgentNotFoundError,
    register_agent_from_yaml_text,
)
from agentflow.services.agent_runner import AgentRunNotFoundError, AgentRunSnapshotMissingError, rerun_agent_run
from agentflow.services.agent_queries import (
    AgentDetail,
    AgentSummary,
    AgentVersionSummary,
    get_registered_agent,
    list_agent_versions,
    list_registered_agents,
)
from agentflow.services.artifact_service import (
    ArtifactFileMissingError,
    ArtifactRecord,
    get_run_artifact,
    list_run_artifacts,
    resolve_artifact_file,
)
from agentflow.services.batch_service import get_batch, list_batches
from agentflow.services.eval_service import (
    get_batch_evaluation_summary,
    list_latest_evaluations_for_runs,
    list_run_evaluations,
)
from agentflow.services.label_service import (
    LabelDuplicateError,
    LabelInvalidError,
    LabelTargetNotFoundError,
    add_run_label,
    add_version_label,
    list_run_labels,
    list_run_labels_for_runs,
    list_version_labels_for_versions,
    remove_run_label,
    remove_version_label,
)
from agentflow.services.preset_service import (
    PresetAgentNotFoundError,
    PresetDuplicateError,
    PresetInvalidError,
    PresetNotFoundError,
    create_input_preset,
    list_input_presets,
    run_from_preset,
)
from agentflow.services.run_actions import (
    RunActionNotFoundError,
    RunRetryNotEligibleError,
    get_manual_retry_eligibility,
    manual_retry_run,
)
from agentflow.services.run_compare import (
    RunCompareInvalidError,
    RunCompareNotFoundError,
    compare_runs,
)
from agentflow.services.run_events import RunEventRecord, list_run_events
from agentflow.services.run_queries import AgentRunDetail, AgentRunSummary, get_agent_run, list_agent_runs
from agentflow.services.stats_queries import DashboardStats, get_dashboard_stats
from agentflow.services.worker_ops import find_stale_runs, list_workers
from agentflow.services.yaml_loader import AgentYamlError

VIEWER_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_LIMIT = 50
MAX_RUN_LIMIT = 200
SETUP_FAILURE_CLASSIFICATIONS = frozenset(
    {
        "config_error",
        "secret_error",
        "provider_setup_error",
        "tool_validation_error",
    }
)

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

    return _render_agent_detail_page(request, parsed_agent_id)


@app.post("/agents/{agent_id}/versions")
async def register_agent_version_action(request: Request, agent_id: str) -> Any:
    parsed_agent_id = _parse_uuid(agent_id)
    if parsed_agent_id is None:
        return _not_found(request, "Agent not found")

    yaml_text = _form_value_from_body(await request.body(), "yaml_text")
    if not yaml_text.strip():
        return _render_agent_detail_page(
            request,
            parsed_agent_id,
            feedback={"message": None, "error": "YAML is required."},
            submitted_yaml=yaml_text,
            status_code=400,
        )

    try:
        result = register_agent_from_yaml_text(yaml_text, agent_id=parsed_agent_id)
    except AgentRegistrationAgentNotFoundError:
        return _not_found(request, "Agent not found")
    except AgentYamlError as exc:
        return _render_agent_detail_page(
            request,
            parsed_agent_id,
            feedback={"message": None, "error": str(exc)},
            submitted_yaml=yaml_text,
            status_code=400,
        )
    except ValidationError as exc:
        return _render_agent_detail_page(
            request,
            parsed_agent_id,
            feedback={"message": None, "error": _format_validation_errors(exc)},
            submitted_yaml=yaml_text,
            status_code=400,
        )
    except ValueError as exc:
        return _render_agent_detail_page(
            request,
            parsed_agent_id,
            feedback={"message": None, "error": str(exc)},
            submitted_yaml=yaml_text,
            status_code=400,
        )

    return _redirect_to_agent(
        parsed_agent_id,
        message=f"New version registered successfully: v{result.version_number}",
    )


@app.post("/agents/{agent_id}/versions/{version_id}/labels")
async def add_version_label_action(request: Request, agent_id: str, version_id: str) -> Any:
    parsed_agent_id = _parse_uuid(agent_id)
    parsed_version_id = _parse_uuid(version_id)
    if parsed_agent_id is None or parsed_version_id is None:
        return _not_found(request, "Agent version not found")

    label = _form_value_from_body(await request.body(), "label")
    try:
        add_version_label(parsed_version_id, label)
    except LabelTargetNotFoundError:
        return _not_found(request, "Agent version not found")
    except (LabelInvalidError, LabelDuplicateError) as exc:
        return _redirect_to_agent(parsed_agent_id, error=str(exc))

    return _redirect_to_agent(parsed_agent_id, message="Version label added")


@app.post("/agents/{agent_id}/versions/{version_id}/labels/remove")
async def remove_version_label_action(request: Request, agent_id: str, version_id: str) -> Any:
    parsed_agent_id = _parse_uuid(agent_id)
    parsed_version_id = _parse_uuid(version_id)
    if parsed_agent_id is None or parsed_version_id is None:
        return _not_found(request, "Agent version not found")

    label = _form_value_from_body(await request.body(), "label")
    try:
        remove_version_label(parsed_version_id, label)
    except LabelTargetNotFoundError:
        return _not_found(request, "Agent version not found")
    except LabelInvalidError as exc:
        return _redirect_to_agent(parsed_agent_id, error=str(exc))

    return _redirect_to_agent(parsed_agent_id, message="Version label removed")


@app.post("/agents/{agent_id}/presets")
async def create_preset_action(request: Request, agent_id: str) -> Any:
    parsed_agent_id = _parse_uuid(agent_id)
    if parsed_agent_id is None:
        return _not_found(request, "Agent not found")

    form = _form_values_from_body(await request.body())
    name = form.get("name", "")
    description = form.get("description", "")
    input_json_text = form.get("input_json", "")
    try:
        input_json = _parse_json_object(input_json_text, label="input_json")
        create_input_preset(
            parsed_agent_id,
            name=name,
            description=description,
            input_json=input_json,
        )
    except PresetAgentNotFoundError:
        return _not_found(request, "Agent not found")
    except (PresetInvalidError, PresetDuplicateError, ValueError) as exc:
        return _render_agent_detail_page(
            request,
            parsed_agent_id,
            feedback={"message": None, "error": str(exc)},
            submitted_preset={
                "name": name,
                "description": description,
                "input_json": input_json_text,
            },
            status_code=400,
        )

    return _redirect_to_agent(parsed_agent_id, message="Input preset created")


def _render_agent_detail_page(
    request: Request,
    agent_id: uuid.UUID,
    *,
    feedback: dict[str, str | None] | None = None,
    submitted_yaml: str = "",
    submitted_preset: dict[str, str] | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    agent = get_registered_agent(agent_id)
    if agent is None:
        return _not_found(request, "Agent not found")

    latest_version_id = agent.latest_version.version_id if agent.latest_version is not None else None
    versions = [
        {
            **_version_summary_to_view(version),
            "is_latest": version.version_id == latest_version_id,
        }
        for version in (list_agent_versions(agent_id) or [])
    ]
    version_labels = list_version_labels_for_versions([
        uuid.UUID(version["version_id"]) for version in versions
    ])
    for version in versions:
        version["labels"] = version_labels.get(uuid.UUID(version["version_id"]), [])

    presets = list_input_presets(agent_id) or []
    return templates.TemplateResponse(
        request,
        "agent_detail.html",
        {
            "page_title": "Agent Detail",
            "agent": _agent_detail_to_view(agent),
            "versions": versions,
            "presets": [_preset_to_view(preset) for preset in presets],
            "feedback": feedback or _feedback_from_request(request),
            "submitted_yaml": submitted_yaml,
            "submitted_preset": submitted_preset or {"name": "", "description": "", "input_json": ""},
        },
        status_code=status_code,
    )


@app.get("/runs", response_class=HTMLResponse)
def runs_index(
    request: Request,
    limit: int = Query(DEFAULT_RUN_LIMIT, ge=1, le=MAX_RUN_LIMIT),
) -> HTMLResponse:
    runs = [_run_summary_to_view(run) for run in list_agent_runs(limit=limit)]
    labels_by_run = list_run_labels_for_runs([uuid.UUID(run["run_id"]) for run in runs])
    for run in runs:
        run["labels"] = labels_by_run.get(uuid.UUID(run["run_id"]), [])
    return templates.TemplateResponse(
        request,
        "runs.html",
        {"page_title": "Runs", "runs": runs, "limit": limit},
    )


@app.get("/batches", response_class=HTMLResponse)
def batches_index(request: Request) -> HTMLResponse:
    batches = [_batch_summary_to_view(batch) for batch in list_batches()]
    return templates.TemplateResponse(
        request,
        "batches.html",
        {"page_title": "Batches", "batches": batches},
    )


@app.get("/ops/workers", response_class=HTMLResponse)
def workers_ops_page(request: Request) -> HTMLResponse:
    stale_threshold_seconds = get_worker_stale_threshold_seconds()
    workers = list_workers(stale_threshold_seconds=stale_threshold_seconds)
    stale_runs = find_stale_runs(stale_threshold_seconds=stale_threshold_seconds)
    active_count = sum(1 for worker in workers if worker.freshness == "active")
    return templates.TemplateResponse(
        request,
        "workers.html",
        {
            "page_title": "Worker Ops",
            "stale_threshold_seconds": stale_threshold_seconds,
            "workers": [_worker_status_to_view(worker) for worker in workers],
            "workers_total": len(workers),
            "workers_active": active_count,
            "workers_stale": len(workers) - active_count,
            "stale_run_candidates": len(stale_runs),
        },
    )


@app.get("/ops/stale-runs", response_class=HTMLResponse)
def stale_runs_ops_page(request: Request) -> HTMLResponse:
    stale_threshold_seconds = get_worker_stale_threshold_seconds()
    stale_runs = find_stale_runs(stale_threshold_seconds=stale_threshold_seconds)
    return templates.TemplateResponse(
        request,
        "stale_runs.html",
        {
            "page_title": "Stale Runs",
            "stale_threshold_seconds": stale_threshold_seconds,
            "stale_runs": [_stale_run_to_view(item) for item in stale_runs],
        },
    )


@app.get("/batches/{batch_id}", response_class=HTMLResponse)
def batch_detail(request: Request, batch_id: str) -> HTMLResponse:
    parsed_batch_id = _parse_uuid(batch_id)
    if parsed_batch_id is None:
        return _not_found(request, "Batch not found")
    batch = get_batch(parsed_batch_id)
    if batch is None:
        return _not_found(request, "Batch not found")
    latest_evaluations = list_latest_evaluations_for_runs([item.run_id for item in batch.items])
    return templates.TemplateResponse(
        request,
        "batch_detail.html",
        {
            "page_title": "Batch Detail",
            "batch": _batch_summary_to_view(batch.summary),
            "items": [
                _batch_item_to_view(item, latest_evaluation=latest_evaluations.get(item.run_id))
                for item in batch.items
            ],
            "evaluation_summary": _batch_evaluation_summary_to_view(get_batch_evaluation_summary(parsed_batch_id)),
        },
    )


@app.get("/runs/compare", response_class=HTMLResponse)
def compare_runs_page(request: Request, run_ids: str = "") -> HTMLResponse:
    parsed_run_ids = [run_id for run_id in (_parse_uuid(value.strip()) for value in run_ids.split(",")) if run_id]
    try:
        comparison = compare_runs(parsed_run_ids)
    except RunCompareNotFoundError as exc:
        return _not_found(request, str(exc))
    except RunCompareInvalidError as exc:
        return _error_page(request, status_code=400, title="Comparison Error", message=str(exc))

    return templates.TemplateResponse(
        request,
        "run_compare.html",
        {
            "page_title": "Compare Runs",
            "agent_id": str(comparison.agent_id),
            "runs": [_comparable_run_to_view(item) for item in comparison.runs],
        },
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
            "labels": [record.label for record in list_run_labels(parsed_run_id)],
            "evaluations": [_evaluation_to_view(evaluation) for evaluation in list_run_evaluations(parsed_run_id)],
            "artifacts": [_artifact_to_view(artifact) for artifact in (list_run_artifacts(parsed_run_id) or [])],
            "feedback": _feedback_from_request(request),
            "can_rerun": bool(run.resolved_config_json),
            "retry_eligible": retry_eligibility.eligible,
            "retry_ineligible_reason": retry_eligibility.reason,
            "is_setup_failure": run.last_error_type in SETUP_FAILURE_CLASSIFICATIONS,
        },
    )


@app.get("/runs/{run_id}/artifacts/{artifact_id}")
def run_artifact(request: Request, run_id: str, artifact_id: str) -> Any:
    parsed_run_id = _parse_uuid(run_id)
    parsed_artifact_id = _parse_uuid(artifact_id)
    if parsed_run_id is None or parsed_artifact_id is None:
        return _not_found(request, "Artifact not found")

    artifact = get_run_artifact(parsed_run_id, parsed_artifact_id)
    if artifact is None:
        return _not_found(request, "Artifact not found")

    try:
        path = resolve_artifact_file(artifact)
    except ArtifactFileMissingError:
        return _error_page(
            request,
            status_code=404,
            title="Artifact File Missing",
            message="The artifact metadata exists, but the local file is missing.",
        )

    return FileResponse(
        path,
        media_type=artifact.mime_type or "application/octet-stream",
        filename=artifact.name,
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


@app.post("/runs/{run_id}/labels")
async def add_run_label_action(request: Request, run_id: str) -> Any:
    parsed_run_id = _parse_uuid(run_id)
    if parsed_run_id is None:
        return _not_found(request, "Run not found")
    label = _form_value_from_body(await request.body(), "label")
    try:
        add_run_label(parsed_run_id, label)
    except LabelTargetNotFoundError:
        return _not_found(request, "Run not found")
    except (LabelInvalidError, LabelDuplicateError) as exc:
        return _redirect_to_run(parsed_run_id, error=str(exc))
    return _redirect_to_run(parsed_run_id, message="Run label added")


@app.post("/runs/{run_id}/labels/remove")
async def remove_run_label_action(request: Request, run_id: str) -> Any:
    parsed_run_id = _parse_uuid(run_id)
    if parsed_run_id is None:
        return _not_found(request, "Run not found")
    label = _form_value_from_body(await request.body(), "label")
    try:
        remove_run_label(parsed_run_id, label)
    except LabelTargetNotFoundError:
        return _not_found(request, "Run not found")
    except LabelInvalidError as exc:
        return _redirect_to_run(parsed_run_id, error=str(exc))
    return _redirect_to_run(parsed_run_id, message="Run label removed")


@app.post("/presets/{preset_id}/run")
def run_preset_action(request: Request, preset_id: str) -> Any:
    parsed_preset_id = _parse_uuid(preset_id)
    if parsed_preset_id is None:
        return _not_found(request, "Preset not found")
    try:
        prepared_run = run_from_preset(parsed_preset_id)
    except PresetNotFoundError:
        return _not_found(request, "Preset not found")
    return _redirect_to_run(prepared_run.run.run_id, message="Run created from preset")


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
        "claimed_by_worker": run.claimed_by_worker,
        "claimed_at": _format_datetime(run.claimed_at),
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
                claimed_by_worker=run.claimed_by_worker,
                claimed_at=run.claimed_at,
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


def _worker_status_to_view(worker: Any) -> dict[str, Any]:
    return {
        "worker_id": str(worker.worker_id),
        "worker_name": worker.worker_name,
        "host": worker.host or "-",
        "pid": worker.pid if worker.pid is not None else "-",
        "status": worker.status,
        "freshness": worker.freshness,
        "heartbeat_age_seconds": worker.heartbeat_age_seconds,
        "last_heartbeat_at": _format_datetime(worker.last_heartbeat_at),
        "started_at": _format_datetime(worker.started_at),
        "updated_at": _format_datetime(worker.updated_at),
    }


def _stale_run_to_view(stale_run: Any) -> dict[str, Any]:
    return {
        "run_id": str(stale_run.run_id),
        "agent_id": str(stale_run.agent_id),
        "claimed_by_worker": stale_run.claimed_by_worker or "-",
        "claimed_at": _format_datetime(stale_run.claimed_at),
        "started_at": _format_datetime(stale_run.started_at),
        "updated_at": _format_datetime(stale_run.updated_at),
        "worker_last_heartbeat_at": _format_datetime(stale_run.worker_last_heartbeat_at),
        "worker_status": stale_run.worker_status or "-",
        "stale_age_seconds": stale_run.stale_age_seconds,
        "reason": stale_run.reason,
    }


def _run_event_to_view(event: RunEventRecord) -> dict[str, Any]:
    return {
        "id": str(event.event_id),
        "event_type": event.event_type,
        "message": event.message,
        "payload_json": _format_json(event.payload_json),
        "created_at": _format_datetime(event.created_at),
    }


def _batch_summary_to_view(batch: Any) -> dict[str, Any]:
    return {
        "batch_id": str(batch.batch_id),
        "agent_id": str(batch.agent_id),
        "version_id": str(batch.version_id),
        "name": batch.name,
        "status": batch.status,
        "item_count": batch.item_count,
        "pending_count": batch.pending_count,
        "running_count": batch.running_count,
        "completed_count": batch.completed_count,
        "failed_count": batch.failed_count,
        "first_started_at": _format_datetime(batch.first_started_at),
        "last_ended_at": _format_datetime(batch.last_ended_at),
        "elapsed_seconds": f"{batch.elapsed_seconds:.3f}" if batch.elapsed_seconds is not None else "-",
        "created_at": _format_datetime(batch.created_at),
        "updated_at": _format_datetime(batch.updated_at),
    }


def _batch_item_to_view(item: Any, *, latest_evaluation: Any | None = None) -> dict[str, Any]:
    return {
        "item_id": str(item.item_id),
        "batch_id": str(item.batch_id),
        "run_id": str(item.run_id),
        "preset_id": str(item.preset_id) if item.preset_id is not None else None,
        "preset_name": item.preset_name,
        "status": item.status,
        "attempt_count": item.attempt_count,
        "max_attempts": item.max_attempts,
        "input_preview": _summarize_json(item.input_json),
        "result_preview": item.result_preview or "-",
        "latest_evaluation": _evaluation_to_view(latest_evaluation) if latest_evaluation is not None else None,
        "started_at": _format_datetime(item.started_at),
        "ended_at": _format_datetime(item.ended_at),
        "created_at": _format_datetime(item.created_at),
    }


def _evaluation_to_view(evaluation: Any) -> dict[str, Any]:
    return {
        "evaluation_id": str(evaluation.evaluation_id),
        "run_id": str(evaluation.run_id),
        "evaluator_type": evaluation.evaluator_type,
        "status": evaluation.status,
        "passed": evaluation.passed,
        "score": evaluation.score,
        "summary": evaluation.summary,
        "expected_json": _format_json(evaluation.expected_json),
        "actual_json": _format_json(evaluation.actual_json),
        "created_at": _format_datetime(evaluation.created_at),
    }


def _artifact_to_view(artifact: ArtifactRecord) -> dict[str, Any]:
    return {
        "artifact_id": str(artifact.artifact_id),
        "run_id": str(artifact.run_id),
        "artifact_type": artifact.artifact_type,
        "name": artifact.name,
        "mime_type": artifact.mime_type or "-",
        "size_bytes": artifact.size_bytes if artifact.size_bytes is not None else "-",
        "description": artifact.description,
        "created_at": _format_datetime(artifact.created_at),
    }


def _batch_evaluation_summary_to_view(summary: Any) -> dict[str, Any]:
    pass_rate = (
        (summary.passed_count / summary.evaluated_count) * 100
        if summary.evaluated_count
        else None
    )
    return {
        "evaluated_count": summary.evaluated_count,
        "passed_count": summary.passed_count,
        "failed_count": summary.failed_count,
        "pass_rate": f"{pass_rate:.1f}%" if pass_rate is not None else "-",
        "latest_created_at": _format_datetime(summary.latest_created_at),
    }


def _preset_to_view(preset: Any) -> dict[str, Any]:
    return {
        "preset_id": str(preset.preset_id),
        "agent_id": str(preset.agent_id),
        "name": preset.name,
        "description": preset.description,
        "input_json": _format_json(preset.input_json),
        "input_preview": _summarize_json(preset.input_json),
        "created_at": _format_datetime(preset.created_at),
        "updated_at": _format_datetime(preset.updated_at),
    }


def _comparable_run_to_view(item: Any) -> dict[str, Any]:
    run = _run_detail_to_view(item.run)
    run["event_count"] = item.event_count
    return run


def _summarize_run_output(output_json: dict[str, Any] | None) -> str | None:
    if output_json is None:
        return None

    parts: list[str] = []
    for key in ("provider_type", "provider", "model", "output_text", "message"):
        value = output_json.get(key)
        if value:
            parts.append(str(value))

    return " | ".join(parts[:3]) if parts else "stored"


def _summarize_json(value: dict[str, Any] | None) -> str:
    if value is None:
        return "-"
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= 80 else f"{text[:77]}..."


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


def _form_value_from_body(body: bytes, key: str) -> str:
    return _form_values_from_body(body).get(key, "")


def _form_values_from_body(body: bytes) -> dict[str, str]:
    form_values = parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True)
    return {key: values[0] if values else "" for key, values in form_values.items()}


def _parse_json_object(value: str, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label}: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid {label}: expected a JSON object.")
    return parsed


def _format_validation_errors(exc: ValidationError) -> str:
    messages: list[str] = []
    for error in exc.errors()[:5]:
        location = ".".join(str(part) for part in error.get("loc", ())) or "config"
        messages.append(f"{location}: {error.get('msg', 'Invalid value')}")

    remaining = len(exc.errors()) - len(messages)
    suffix = f" ({remaining} more)" if remaining > 0 else ""
    return f"Validation failed: {'; '.join(messages)}{suffix}"


def _feedback_from_request(request: Request) -> dict[str, str | None]:
    return {
        "message": request.query_params.get("message"),
        "error": request.query_params.get("error"),
    }


def _redirect_to_agent(agent_id: uuid.UUID, *, message: str | None = None, error: str | None = None) -> RedirectResponse:
    target = f"/agents/{agent_id}"
    query_parts: list[str] = []
    if message:
        query_parts.append(f"message={quote(message, safe='')}")
    if error:
        query_parts.append(f"error={quote(error, safe='')}")
    if query_parts:
        target = f"{target}?{'&'.join(query_parts)}"
    return RedirectResponse(target, status_code=303)


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
