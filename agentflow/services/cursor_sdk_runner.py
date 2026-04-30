from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from sqlalchemy.orm import Session, sessionmaker

from agentflow.db.session import create_session_factory
from agentflow.services.external_runner import (
    ExternalRunnerCommitError,
    ExternalRunnerError,
    ExternalRunnerGitError,
    commit_all_changes,
    get_changed_files_for_commit,
    get_head_commit,
    has_worktree_changes,
    resolve_runner_cwd,
    run_git,
)
from agentflow.services.run_code_changes import RunCodeChangeRecord, create_run_code_change
from agentflow.services.run_events import (
    RUN_EVENT_CURSOR_BRIDGE_EVENT,
    RUN_EVENT_CURSOR_RUNNER_COMPLETED,
    RUN_EVENT_CURSOR_RUNNER_FAILED,
    RUN_EVENT_CURSOR_RUNNER_STARTED,
    RUN_EVENT_GIT_CHANGES_COMMITTED,
    RUN_EVENT_GIT_CHANGES_DETECTED,
    RUN_EVENT_GIT_NO_CHANGES,
    record_run_event,
)
from agentflow.services.runtime_validation import RuntimeValidationError, validate_run_configuration
from agentflow.services.secret_resolution import SecretResolutionError, resolve_secret_ref

CURSOR_BRIDGE_PATH_ENV = "AGENTFLOW_CURSOR_BRIDGE_PATH"
DEFAULT_STDERR_TAIL_CHARS = 4000


class CursorSdkRunnerError(ExternalRunnerError):
    """Base error for Cursor SDK runner execution."""


class CursorSdkRunnerBridgeError(CursorSdkRunnerError):
    pass


@dataclass(frozen=True)
class CursorSdkExecutionResult:
    exit_code: int
    stderr_tail: str
    base_commit_sha: str
    result_commit_sha: str
    commit_message: str
    changed_files: list[dict[str, str]]
    model: str
    runtime: str
    code_change_record: RunCodeChangeRecord


@dataclass(frozen=True)
class _BridgeProcessResult:
    exit_code: int
    stdout_event_count: int
    stderr_text: str
    timed_out: bool


def execute_cursor_sdk_runner(
    run_id: uuid.UUID,
    resolved_config_json: dict[str, Any],
    *,
    input_json: dict[str, Any] | None,
    base_dir: Path | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> CursorSdkExecutionResult:
    session_factory = session_factory or create_session_factory()
    commit_message = f"agentflow run {run_id}"

    try:
        validated = validate_run_configuration(resolved_config_json)
        runner = validated.runner
        if runner is None or runner.runner_type != "cursor_sdk":
            raise CursorSdkRunnerError("Run does not define runner.type=cursor_sdk.")

        if runner.runtime != "local":
            raise CursorSdkRunnerError("runner.runtime must be local for cursor_sdk in this phase.")

        if runner.api_key_ref is None:
            raise CursorSdkRunnerError("runner.api_key_ref is required for cursor_sdk.")

        try:
            api_key = resolve_secret_ref(runner.api_key_ref, env=os.environ)
        except SecretResolutionError as exc:
            raise CursorSdkRunnerError(str(exc)) from exc

        resolved_cwd = resolve_runner_cwd(runner.cwd, base_dir=base_dir)
        _ensure_git_repository(resolved_cwd)

        base_commit_sha = get_head_commit(resolved_cwd)
        model = runner.model or "auto"
        runtime = runner.runtime or "local"

        record_run_event(
            run_id,
            event_type=RUN_EVENT_CURSOR_RUNNER_STARTED,
            message="Cursor SDK runner execution started.",
            payload_json={
                "runner_type": "cursor_sdk",
                "runtime": runtime,
                "model": model,
                "cwd": str(resolved_cwd),
                "timeout_seconds": runner.timeout_seconds,
                "base_commit_sha": base_commit_sha,
            },
            session_factory=session_factory,
        )

        prompt = _build_cursor_prompt(
            system_prompt=_extract_system_prompt(resolved_config_json),
            input_json=input_json,
        )

        bridge_result = _run_cursor_bridge(
            run_id=run_id,
            cwd=resolved_cwd,
            api_key=api_key,
            prompt=prompt,
            model=model,
            timeout_seconds=runner.timeout_seconds,
            resolved_config_json=resolved_config_json,
            session_factory=session_factory,
        )

        result_commit_sha = base_commit_sha
        changed_files: list[dict[str, str]] = []
        if has_worktree_changes(resolved_cwd):
            record_run_event(
                run_id,
                event_type=RUN_EVENT_GIT_CHANGES_DETECTED,
                message="Git worktree changes detected after Cursor SDK execution.",
                payload_json={"cwd": str(resolved_cwd)},
                session_factory=session_factory,
            )
            result_commit_sha = commit_all_changes(resolved_cwd, commit_message)
            changed_files = get_changed_files_for_commit(resolved_cwd, result_commit_sha)
            record_run_event(
                run_id,
                event_type=RUN_EVENT_GIT_CHANGES_COMMITTED,
                message="Git worktree changes committed.",
                payload_json={
                    "result_commit_sha": result_commit_sha,
                    "changed_files_count": len(changed_files),
                },
                session_factory=session_factory,
            )
        else:
            record_run_event(
                run_id,
                event_type=RUN_EVENT_GIT_NO_CHANGES,
                message="No git worktree changes detected after Cursor SDK execution.",
                payload_json={"result_commit_sha": result_commit_sha},
                session_factory=session_factory,
            )

        code_change_record = create_run_code_change(
            run_id,
            runner_type="cursor_sdk",
            base_commit_sha=base_commit_sha,
            result_commit_sha=result_commit_sha,
            commit_message=commit_message,
            changed_files_json=changed_files,
            session_factory=session_factory,
        )

        stderr_tail = _tail_text(bridge_result.stderr_text, DEFAULT_STDERR_TAIL_CHARS)

        if bridge_result.exit_code == 0:
            record_run_event(
                run_id,
                event_type=RUN_EVENT_CURSOR_RUNNER_COMPLETED,
                message="Cursor SDK runner execution completed.",
                payload_json={
                    "exit_code": bridge_result.exit_code,
                    "cursor_events": bridge_result.stdout_event_count,
                    "base_commit_sha": base_commit_sha,
                    "result_commit_sha": result_commit_sha,
                    "changed_files_count": len(changed_files),
                },
                session_factory=session_factory,
            )
        else:
            failure_message = "Cursor SDK runner exited with a non-zero status."
            if bridge_result.timed_out:
                failure_message = "Cursor SDK runner timed out."

            record_run_event(
                run_id,
                event_type=RUN_EVENT_CURSOR_RUNNER_FAILED,
                message=failure_message,
                payload_json={
                    "exit_code": bridge_result.exit_code,
                    "timed_out": bridge_result.timed_out,
                    "cursor_events": bridge_result.stdout_event_count,
                    "base_commit_sha": base_commit_sha,
                    "result_commit_sha": result_commit_sha,
                    "changed_files_count": len(changed_files),
                    "stderr_tail": stderr_tail,
                },
                session_factory=session_factory,
            )

        return CursorSdkExecutionResult(
            exit_code=bridge_result.exit_code,
            stderr_tail=stderr_tail,
            base_commit_sha=base_commit_sha,
            result_commit_sha=result_commit_sha,
            commit_message=commit_message,
            changed_files=changed_files,
            model=model,
            runtime=runtime,
            code_change_record=code_change_record,
        )
    except (RuntimeValidationError, CursorSdkRunnerError, ExternalRunnerGitError, ExternalRunnerCommitError) as exc:
        record_run_event(
            run_id,
            event_type=RUN_EVENT_CURSOR_RUNNER_FAILED,
            message="Cursor SDK runner failed before completion.",
            payload_json={"error": str(exc)},
            session_factory=session_factory,
        )
        if isinstance(exc, CursorSdkRunnerError):
            raise
        raise CursorSdkRunnerError(str(exc)) from exc


def _run_cursor_bridge(
    run_id: uuid.UUID,
    cwd: Path,
    api_key: str,
    prompt: str,
    model: str,
    timeout_seconds: int,
    resolved_config_json: dict[str, Any],
    session_factory: sessionmaker[Session],
) -> _BridgeProcessResult:
    bridge_path, is_override = _resolve_bridge_path()
    if not bridge_path.is_file():
        raise CursorSdkRunnerBridgeError(
            f"Cursor SDK bridge script not found: {bridge_path}. "
            f"Set {CURSOR_BRIDGE_PATH_ENV} to a valid executable script path."
        )

    command = _build_bridge_command(bridge_path, is_override=is_override)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    request_payload = {
        "apiKey": api_key,
        "cwd": str(cwd),
        "prompt": prompt,
        "model": model,
        "metadata": {
            "runId": str(run_id),
            "agentName": _extract_agent_name(resolved_config_json),
            "agentVersion": _extract_agent_version(resolved_config_json),
        },
    }

    stdout_queue: queue.Queue[str] = queue.Queue()
    stderr_queue: queue.Queue[str] = queue.Queue()
    stdout_done = threading.Event()
    stderr_done = threading.Event()

    stdout_thread = threading.Thread(
        target=_read_stream_lines,
        args=(process.stdout, stdout_queue, stdout_done),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_read_stream_lines,
        args=(process.stderr, stderr_queue, stderr_done),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        assert process.stdin is not None
        process.stdin.write(json.dumps(request_payload) + "\n")
        process.stdin.flush()
        process.stdin.close()
    except Exception as exc:
        process.kill()
        raise CursorSdkRunnerBridgeError(f"Failed to send request to Cursor bridge: {exc}") from exc

    start = time.monotonic()
    stdout_event_count = 0
    stderr_lines: list[str] = []
    timed_out = False

    while True:
        if time.monotonic() - start > timeout_seconds:
            timed_out = True
            process.kill()
            break

        _drain_text_queue(stderr_queue, stderr_lines)

        had_stdout = _drain_stdout_events(
            run_id,
            stdout_queue=stdout_queue,
            session_factory=session_factory,
        )
        stdout_event_count += had_stdout

        if process.poll() is not None and stdout_done.is_set() and stderr_done.is_set() and stdout_queue.empty() and stderr_queue.empty():
            break

        time.sleep(0.05)

    _drain_text_queue(stderr_queue, stderr_lines)
    stdout_event_count += _drain_stdout_events(
        run_id,
        stdout_queue=stdout_queue,
        session_factory=session_factory,
    )

    returncode = process.wait()
    if timed_out:
        returncode = 124

    return _BridgeProcessResult(
        exit_code=returncode,
        stdout_event_count=stdout_event_count,
        stderr_text="".join(stderr_lines),
        timed_out=timed_out,
    )


def _drain_stdout_events(
    run_id: uuid.UUID,
    *,
    stdout_queue: queue.Queue[str],
    session_factory: sessionmaker[Session],
) -> int:
    count = 0
    while True:
        try:
            line = stdout_queue.get_nowait()
        except queue.Empty:
            break

        normalized = line.strip()
        if not normalized:
            continue

        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            payload = {"parse_error": True, "raw_line": normalized}

        record_run_event(
            run_id,
            event_type=RUN_EVENT_CURSOR_BRIDGE_EVENT,
            message="Cursor bridge event received.",
            payload_json={"event": payload},
            session_factory=session_factory,
        )
        count += 1

    return count


def _read_stream_lines(
    stream: TextIO | None,
    target_queue: queue.Queue[str],
    done: threading.Event,
) -> None:
    if stream is None:
        done.set()
        return

    try:
        for line in stream:
            target_queue.put(line)
    finally:
        done.set()


def _drain_text_queue(source_queue: queue.Queue[str], output_lines: list[str]) -> None:
    while True:
        try:
            output_lines.append(source_queue.get_nowait())
        except queue.Empty:
            return


def _resolve_bridge_path() -> tuple[Path, bool]:
    configured = os.environ.get(CURSOR_BRIDGE_PATH_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve(), True

    return (
        Path(__file__).resolve().parents[2] / "agentflow_js" / "cursor_runner" / "src" / "cursor_runner.mjs",
        False,
    )


def _build_bridge_command(bridge_path: Path, *, is_override: bool) -> list[str]:
    suffix = bridge_path.suffix.lower()
    if suffix in {".js", ".mjs", ".cjs"} or not is_override:
        node_binary = shutil.which("node")
        if node_binary is None:
            raise CursorSdkRunnerBridgeError("Node.js is required for cursor_sdk runner, but 'node' is not installed.")
        return [node_binary, str(bridge_path)]

    if suffix == ".py":
        return [sys.executable, str(bridge_path)]

    return [str(bridge_path)]


def _build_cursor_prompt(*, system_prompt: str, input_json: dict[str, Any] | None) -> str:
    parts = [system_prompt.strip()]
    if input_json:
        parts.append("Run input JSON:\n" + json.dumps(input_json, indent=2, sort_keys=True))
    return "\n\n".join(part for part in parts if part)


def _extract_system_prompt(resolved_config_json: dict[str, Any]) -> str:
    value = resolved_config_json.get("system_prompt")
    if isinstance(value, str) and value.strip():
        return value
    return "You are a coding agent running through Cursor SDK."


def _extract_agent_name(resolved_config_json: dict[str, Any]) -> str:
    value = resolved_config_json.get("name")
    if isinstance(value, str) and value.strip():
        return value
    return "agentflow-run"


def _extract_agent_version(resolved_config_json: dict[str, Any]) -> int:
    value = resolved_config_json.get("version")
    if isinstance(value, int) and value > 0:
        return value
    return 1


def _ensure_git_repository(cwd: Path) -> None:
    completed = run_git(["rev-parse", "--is-inside-work-tree"], cwd)
    if completed.stdout.strip().lower() != "true":
        raise ExternalRunnerGitError(f"Path is not inside a git worktree: {cwd}")


def _tail_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[-max_chars:]
