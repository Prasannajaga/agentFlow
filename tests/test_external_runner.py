from __future__ import annotations

import subprocess

import pytest

from agentflow.services.agent_registry import register_agent_from_yaml_text
from agentflow.services.agent_runner import create_run_for_agent, execute_claimed_run
from agentflow.services.external_runner import get_head_commit
from agentflow.services.run_code_changes import get_latest_run_code_change
from agentflow.services.run_queries import claim_next_pending_run
from conftest import make_agent_yaml


def test_git_helper_get_head_commit(tmp_path) -> None:
    repo_path = _init_git_repo(tmp_path)

    head_commit = get_head_commit(repo_path)
    expected = _run(["git", "rev-parse", "HEAD"], cwd=repo_path).stdout.strip()

    assert head_commit
    assert head_commit == expected


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
    prepared = create_run_for_agent(registration.agent_id, session_factory=db_session_factory)
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
    assert code_change.base_commit_sha == completed.output_json["base_commit_sha"]
    assert code_change.result_commit_sha == completed.output_json["result_commit_sha"]
    assert code_change.commit_message == f"agentflow run {completed.run_id}"
    assert any(
        item["path"] == "generated.txt" and item["status"] == "added"
        for item in code_change.changed_files_json
    )


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
