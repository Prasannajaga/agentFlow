from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from agentflow.db.session import create_session_factory
from agentflow.services.run_code_changes import RunCodeChangeRecord, create_run_code_change
from agentflow.services.run_events import (
    RUN_EVENT_EXTERNAL_RUNNER_COMPLETED,
    RUN_EVENT_EXTERNAL_RUNNER_FAILED,
    RUN_EVENT_EXTERNAL_RUNNER_STARTED,
    record_run_event,
)
from agentflow.services.runtime_validation import RuntimeValidationError, validate_run_configuration


class ExternalRunnerError(RuntimeError):
    """Base error for external CLI runner execution."""


class ExternalRunnerGitError(ExternalRunnerError):
    pass


class ExternalRunnerCommitError(ExternalRunnerError):
    pass


@dataclass(frozen=True)
class ExternalRunnerExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    base_commit_sha: str | None
    result_commit_sha: str | None
    commit_message: str
    changed_files: list[dict[str, str]]
    code_change_record: RunCodeChangeRecord


def run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or "git command failed"
        raise ExternalRunnerGitError(f"git {' '.join(args)} failed: {detail}")
    return completed


def get_head_commit(cwd: Path) -> str:
    completed = run_git(["rev-parse", "HEAD"], cwd)
    return completed.stdout.strip()


def has_worktree_changes(cwd: Path) -> bool:
    completed = run_git(["status", "--porcelain"], cwd)
    return bool(completed.stdout.strip())


def commit_all_changes(cwd: Path, message: str) -> str:
    run_git(["add", "-A"], cwd)
    try:
        run_git(["commit", "-m", message], cwd)
    except ExternalRunnerGitError as exc:
        raise ExternalRunnerCommitError(str(exc)) from exc
    return get_head_commit(cwd)


def get_changed_files_for_commit(cwd: Path, commit_sha: str) -> list[dict[str, str]]:
    completed = run_git(["diff-tree", "--no-commit-id", "--name-status", "-r", commit_sha], cwd)
    changed_files: list[dict[str, str]] = []

    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part for part in line.split("\t") if part]
        if len(parts) < 2:
            continue

        status_token = parts[0]
        path = parts[-1]
        status = _map_git_status(status_token)
        changed_files.append({"path": path, "status": status})

    return changed_files


def execute_external_cli_runner(
    run_id: uuid.UUID,
    resolved_config_json: dict[str, Any],
    *,
    base_dir: Path | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> ExternalRunnerExecutionResult:
    session_factory = session_factory or create_session_factory()
    commit_message = f"agentflow run {run_id}"

    try:
        validated = validate_run_configuration(resolved_config_json)
        runner = validated.runner
        if runner is None or runner.runner_type != "external_cli":
            raise ExternalRunnerError("Run does not define runner.type=external_cli.")

        resolved_cwd = resolve_runner_cwd(runner.cwd, base_dir=base_dir)
        _ensure_git_repository(resolved_cwd)

        if has_worktree_changes(resolved_cwd):
            raise ExternalRunnerGitError(
                f"External runner requires a clean git worktree: {resolved_cwd}"
            )

        base_commit_sha = get_head_commit(resolved_cwd)
        command = [runner.command, *runner.args]

        record_run_event(
            run_id,
            event_type=RUN_EVENT_EXTERNAL_RUNNER_STARTED,
            message="External CLI runner execution started.",
            payload_json={
                "command": command,
                "cwd": str(resolved_cwd),
                "timeout_seconds": runner.timeout_seconds,
                "base_commit_sha": base_commit_sha,
            },
            session_factory=session_factory,
        )

        exit_code, stdout, stderr = _run_external_command(
            command,
            cwd=resolved_cwd,
            timeout_seconds=runner.timeout_seconds,
        )

        result_commit_sha = base_commit_sha
        changed_files: list[dict[str, str]] = []
        if has_worktree_changes(resolved_cwd):
            result_commit_sha = commit_all_changes(resolved_cwd, commit_message)
            changed_files = get_changed_files_for_commit(resolved_cwd, result_commit_sha)

        code_change_record = create_run_code_change(
            run_id,
            base_commit_sha=base_commit_sha,
            result_commit_sha=result_commit_sha,
            commit_message=commit_message,
            changed_files_json=changed_files,
            session_factory=session_factory,
        )

        if exit_code == 0:
            record_run_event(
                run_id,
                event_type=RUN_EVENT_EXTERNAL_RUNNER_COMPLETED,
                message="External CLI runner execution completed.",
                payload_json={
                    "exit_code": exit_code,
                    "base_commit_sha": base_commit_sha,
                    "result_commit_sha": result_commit_sha,
                    "changed_files_count": len(changed_files),
                },
                session_factory=session_factory,
            )
        else:
            record_run_event(
                run_id,
                event_type=RUN_EVENT_EXTERNAL_RUNNER_FAILED,
                message="External CLI runner exited with a non-zero status.",
                payload_json={
                    "exit_code": exit_code,
                    "base_commit_sha": base_commit_sha,
                    "result_commit_sha": result_commit_sha,
                    "changed_files_count": len(changed_files),
                },
                session_factory=session_factory,
            )

        return ExternalRunnerExecutionResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            base_commit_sha=base_commit_sha,
            result_commit_sha=result_commit_sha,
            commit_message=commit_message,
            changed_files=changed_files,
            code_change_record=code_change_record,
        )
    except (RuntimeValidationError, ExternalRunnerError) as exc:
        record_run_event(
            run_id,
            event_type=RUN_EVENT_EXTERNAL_RUNNER_FAILED,
            message="External CLI runner failed before completion.",
            payload_json={"error": str(exc)},
            session_factory=session_factory,
        )
        if isinstance(exc, ExternalRunnerError):
            raise
        raise ExternalRunnerError(str(exc)) from exc


def resolve_runner_cwd(runner_cwd: str, *, base_dir: Path | None = None) -> Path:
    base_path = (base_dir or Path.cwd()).resolve()
    resolved = (base_path / runner_cwd).resolve()

    if not resolved.is_dir():
        raise ExternalRunnerError(f"Runner cwd does not exist or is not a directory: {resolved}")

    try:
        resolved.relative_to(base_path)
    except ValueError as exc:
        raise ExternalRunnerError(
            f"Runner cwd escapes the allowed base directory: {runner_cwd}"
        ) from exc

    return resolved


def _ensure_git_repository(cwd: Path) -> None:
    completed = run_git(["rev-parse", "--is-inside-work-tree"], cwd)
    if completed.stdout.strip().lower() != "true":
        raise ExternalRunnerGitError(f"Path is not inside a git worktree: {cwd}")


def _run_external_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        return completed.returncode, completed.stdout or "", completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        stdout = _normalize_process_output(exc.stdout)
        stderr = _normalize_process_output(exc.stderr)
        timeout_message = f"Command timed out after {timeout_seconds} seconds."
        stderr = f"{stderr}\n{timeout_message}" if stderr else timeout_message
        return 124, stdout, stderr
    except FileNotFoundError as exc:
        raise ExternalRunnerError(f"Runner command not found: {command[0]}") from exc


def _normalize_process_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _map_git_status(status_token: str) -> str:
    status_code = status_token[:1]
    if status_code == "A":
        return "added"
    if status_code == "M":
        return "modified"
    if status_code == "D":
        return "deleted"
    if status_code == "R":
        return "renamed"
    if status_code == "C":
        return "copied"
    return status_token
