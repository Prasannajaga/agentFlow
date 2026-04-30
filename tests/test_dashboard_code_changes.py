from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from agentflow.services.run_actions import RetryEligibility
from agentflow.services.run_code_changes import RunCodeChangeRecord
from agentflow.services.run_queries import AgentRunDetail
from agentflow.viewer import main as viewer_main


def test_run_detail_renders_code_changes(monkeypatch) -> None:
    run_id = uuid.uuid4()
    run = _build_run(run_id)
    code_change = RunCodeChangeRecord(
        code_change_id=uuid.uuid4(),
        run_id=run_id,
        base_commit_sha="abc123",
        result_commit_sha="def456",
        commit_message=f"agentflow run {run_id}",
        changed_files_json=[{"path": "generated.txt", "status": "added"}],
        created_at=datetime.now(timezone.utc),
    )
    _patch_run_detail_dependencies(monkeypatch, run=run, code_change=code_change)

    client = TestClient(viewer_main.app)
    response = client.get(f"/runs/{run_id}")

    assert response.status_code == 200
    assert "Code Changes" in response.text
    assert "abc123" in response.text
    assert "def456" in response.text
    assert "generated.txt" in response.text


def test_run_detail_renders_no_code_changes_message(monkeypatch) -> None:
    run_id = uuid.uuid4()
    run = _build_run(run_id)
    _patch_run_detail_dependencies(monkeypatch, run=run, code_change=None)

    client = TestClient(viewer_main.app)
    response = client.get(f"/runs/{run_id}")

    assert response.status_code == 200
    assert "No code changes recorded for this run." in response.text


def _patch_run_detail_dependencies(monkeypatch, *, run: AgentRunDetail, code_change: RunCodeChangeRecord | None) -> None:
    monkeypatch.setattr(viewer_main, "get_agent_run", lambda _run_id: run)
    monkeypatch.setattr(viewer_main, "list_run_events", lambda _run_id: [])
    monkeypatch.setattr(
        viewer_main,
        "get_manual_retry_eligibility",
        lambda _run: RetryEligibility(eligible=True, reason=None),
    )
    monkeypatch.setattr(viewer_main, "list_run_labels", lambda _run_id: [])
    monkeypatch.setattr(viewer_main, "list_run_evaluations", lambda _run_id: [])
    monkeypatch.setattr(viewer_main, "list_run_artifacts", lambda _run_id: [])
    monkeypatch.setattr(viewer_main, "get_latest_run_code_change", lambda _run_id: code_change)


def _build_run(run_id: uuid.UUID) -> AgentRunDetail:
    now = datetime.now(timezone.utc)
    return AgentRunDetail(
        run_id=run_id,
        agent_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        source_run_id=None,
        status="completed",
        input_json={"prompt": "hello"},
        resolved_config_json={"name": "demo"},
        output_json={"runner_type": "external_cli", "exit_code": 0},
        error_message=None,
        last_error_type=None,
        attempt_count=1,
        max_attempts=1,
        retryable=True,
        created_at=now,
        started_at=now,
        ended_at=now,
        updated_at=now,
        claimed_by_worker="worker",
        claimed_at=now,
    )
