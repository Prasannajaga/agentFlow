from __future__ import annotations

import subprocess

import pytest

from agentflow.services.agent_registry import register_agent_from_yaml_text
from agentflow.services.agent_runner import (
    AgentRunExecutionFailedError,
    create_run_for_agent,
    execute_claimed_run,
)
from agentflow.services.external_runner import (
    commit_all_changes,
    get_changed_files_for_commit,
    get_head_commit,
)
from agentflow.services.run_code_changes import get_latest_run_code_change
from agentflow.services.run_events import list_run_events
from agentflow.services.run_queries import claim_next_pending_run, get_agent_run
from conftest import make_agent_yaml


def test_git_helper_get_head_commit(tmp_path) -> None:
    repo_path = _init_git_repo(tmp_path)

    head_commit = get_head_commit(repo_path)
    expected = _run(["git", "rev-parse", "HEAD"], cwd=repo_path).stdout.strip()

    assert head_commit
    assert head_commit == expected


def test_git_helper_commit_records_created_and_modified_files(tmp_path) -> None:
    repo_path = _init_git_repo(tmp_path)

    readme = repo_path / "README.md"
    readme.write_text("# temp repo updated\n", encoding="utf-8")
    created = repo_path / "new_file.txt"
    created.write_text("new\n", encoding="utf-8")

    commit_sha = commit_all_changes(repo_path, "test commit")
    changed_files = get_changed_files_for_commit(repo_path, commit_sha)

    assert any(item["path"] == "README.md" and item["status"] == "modified" for item in changed_files)
    assert any(item["path"] == "new_file.txt" and item["status"] == "added" for item in changed_files)


@pytest.mark.db
def test_external_runner_commits_changes_and_stores_output(monkeypatch: pytest.MonkeyPatch, tmp_path, db_session_factory) -> None:
    repo_path = _init_git_repo(tmp_path)
    monkeypatch.chdir(repo_path)

    raw_yaml = make_agent_yaml(
        name="external-runner-agent",
        extra={
            "provider": None,
            "runner": {
                "type": "external_cli",
                "command": "python",
                "args": [
                    "-c",
                    "from pathlib import Path; Path('generated.txt').write_text('hello from runner\\n', encoding='utf-8')",
                ],
                "cwd": ".",
                "timeout_seconds": 30,
            },
            "runtime": {
                "timeout_seconds": 30,
                "retry": {
                    "max_attempts": 1,
                    "backoff_seconds": 0,
                },
            },
        },
    )

    registration = register_agent_from_yaml_text(raw_yaml, session_factory=db_session_factory)
    _prepared = create_run_for_agent(registration.agent_id, session_factory=db_session_factory)
    claimed = claim_next_pending_run(worker_id="external-worker", session_factory=db_session_factory)

    assert claimed is not None

    completed = execute_claimed_run(claimed, session_factory=db_session_factory)

    assert completed.status == "completed"
    assert completed.output_json is not None
    assert completed.output_json["runner_type"] == "external_cli"
    assert completed.output_json["exit_code"] == 0
    assert completed.output_json["base_commit_sha"]
    assert completed.output_json["result_commit_sha"]

    changed_files = completed.output_json["changed_files"]
    assert changed_files
    assert any(item["path"] == "generated.txt" and item["status"] == "added" for item in changed_files)

    code_change = get_latest_run_code_change(completed.run_id, session_factory=db_session_factory)
    assert code_change is not None
    assert code_change.runner_type == "external_cli"
    assert code_change.base_commit_sha == completed.output_json["base_commit_sha"]
    assert code_change.result_commit_sha == completed.output_json["result_commit_sha"]
    assert code_change.commit_message == f"agentflow run {completed.run_id}"
    assert any(
        item["path"] == "generated.txt" and item["status"] == "added"
        for item in code_change.changed_files_json
    )


@pytest.mark.db
def test_cursor_runner_fake_bridge_streams_events_and_commits_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    db_session_factory,
) -> None:
    repo_path = _init_git_repo(tmp_path)
    monkeypatch.chdir(repo_path)
    bridge_path = _write_fake_cursor_bridge(tmp_path)

    monkeypatch.setenv("CURSOR_API_KEY", "test-cursor-key")
    monkeypatch.setenv("AGENTFLOW_CURSOR_BRIDGE_PATH", str(bridge_path))

    raw_yaml = make_agent_yaml(
        name="cursor-sdk-agent",
        extra={
            "provider": {
                "type": "fake",
                "model": "stub-model",
            },
            "runner": {
                "type": "cursor_sdk",
                "runtime": "local",
                "model": "auto",
                "cwd": ".",
                "api_key_ref": "env:CURSOR_API_KEY",
                "timeout_seconds": 30,
            },
            "runtime": {
                "timeout_seconds": 30,
                "retry": {
                    "max_attempts": 1,
                    "backoff_seconds": 0,
                },
            },
        },
    )

    registration = register_agent_from_yaml_text(raw_yaml, session_factory=db_session_factory)
    _prepared = create_run_for_agent(registration.agent_id, session_factory=db_session_factory)
    claimed = claim_next_pending_run(worker_id="cursor-worker", session_factory=db_session_factory)
    assert claimed is not None

    completed = execute_claimed_run(claimed, session_factory=db_session_factory)

    assert completed.status == "completed"
    assert completed.output_json is not None
    assert completed.output_json["runner_type"] == "cursor_sdk"
    assert completed.output_json["exit_code"] == 0
    assert completed.output_json["runtime"] == "local"
    assert completed.output_json["model"] == "auto"
    assert completed.output_json["base_commit_sha"]
    assert completed.output_json["result_commit_sha"]
    assert completed.output_json["commit_message"] == f"agentflow run {completed.run_id}"
    assert completed.output_json["stderr_tail"] == ""
    changed_files = completed.output_json["changed_files"]
    assert any(item["path"] == "cursor_generated.txt" and item["status"] == "added" for item in changed_files)

    events = list_run_events(completed.run_id, session_factory=db_session_factory)
    event_types = [event.event_type for event in events]
    assert "cursor_runner_started" in event_types
    assert "cursor_bridge_event" in event_types
    assert "git_changes_detected" in event_types
    assert "git_changes_committed" in event_types
    assert "cursor_runner_completed" in event_types

    code_change = get_latest_run_code_change(completed.run_id, session_factory=db_session_factory)
    assert code_change is not None
    assert code_change.runner_type == "cursor_sdk"
    assert code_change.base_commit_sha == completed.output_json["base_commit_sha"]
    assert code_change.result_commit_sha == completed.output_json["result_commit_sha"]
    assert any(
        item["path"] == "cursor_generated.txt" and item["status"] == "added"
        for item in code_change.changed_files_json
    )


@pytest.mark.db
def test_cursor_runner_missing_api_key_fails_run(monkeypatch: pytest.MonkeyPatch, tmp_path, db_session_factory) -> None:
    repo_path = _init_git_repo(tmp_path)
    monkeypatch.chdir(repo_path)
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)

    raw_yaml = make_agent_yaml(
        name="cursor-sdk-missing-key",
        extra={
            "provider": {
                "type": "fake",
                "model": "stub-model",
            },
            "runner": {
                "type": "cursor_sdk",
                "runtime": "local",
                "model": "auto",
                "cwd": ".",
                "api_key_ref": "env:CURSOR_API_KEY",
                "timeout_seconds": 30,
            },
        },
    )

    registration = register_agent_from_yaml_text(raw_yaml, session_factory=db_session_factory)
    prepared = create_run_for_agent(registration.agent_id, session_factory=db_session_factory)
    claimed = claim_next_pending_run(worker_id="cursor-worker", session_factory=db_session_factory)
    assert claimed is not None

    with pytest.raises(AgentRunExecutionFailedError):
        execute_claimed_run(claimed, session_factory=db_session_factory)

    failed = get_agent_run(prepared.run.run_id, session_factory=db_session_factory)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_message is not None
    assert "CURSOR_API_KEY" in failed.error_message


def _init_git_repo(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir(parents=True, exist_ok=True)

    _run(["git", "init"], cwd=repo_path)
    _run(["git", "config", "user.email", "tests@example.com"], cwd=repo_path)
    _run(["git", "config", "user.name", "AgentFlow Tests"], cwd=repo_path)

    readme = repo_path / "README.md"
    readme.write_text("# temp repo\n", encoding="utf-8")
    _run(["git", "add", "README.md"], cwd=repo_path)
    _run(["git", "commit", "-m", "initial"], cwd=repo_path)

    return repo_path


def _run(command: list[str], *, cwd):
    return subprocess.run(command, cwd=str(cwd), check=True, capture_output=True, text=True)


def _write_fake_cursor_bridge(tmp_path):
    bridge_path = tmp_path / "fake_cursor_bridge.py"
    bridge_path.write_text(
        """
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read())
cwd = Path(request["cwd"])
(cwd / "cursor_generated.txt").write_text("generated by fake bridge\\n", encoding="utf-8")


def emit(payload):
    sys.stdout.write(json.dumps(payload) + "\\n")
    sys.stdout.flush()


emit({"type": "bridge_started", "payload": {"cwd": request["cwd"]}})
emit({"type": "cursor_event", "payload": {"kind": "token", "text": "ok"}})
emit({"type": "cursor_completed", "payload": {"status": "completed"}})
emit({"type": "bridge_completed", "payload": {}})
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return bridge_path
