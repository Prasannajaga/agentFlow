from __future__ import annotations

import os
from typing import Any

FAKE_PROVIDER_NAME = "fake"
FAKE_MODEL_NAME = "stub-model"
FAKE_FAILURE_ENV = "AGENTFLOW_FAKE_PROVIDER_FAIL"


def execute_fake_agent(config: dict[str, Any]) -> dict[str, Any]:
    if _should_fail():
        raise RuntimeError("Configured fake provider failure.")

    task_config = config.get("task")
    task_type = task_config.get("type") if isinstance(task_config, dict) else None

    tools = config.get("tools")
    normalized_tools = list(tools) if isinstance(tools, list) else []

    return {
        "provider": FAKE_PROVIDER_NAME,
        "model": FAKE_MODEL_NAME,
        "message": "Synchronous fake run completed successfully.",
        "agent_name": config.get("name"),
        "system_prompt_preview": _preview_text(config.get("system_prompt")),
        "tools": normalized_tools,
        "task_type": task_type,
    }


def _should_fail() -> bool:
    return os.environ.get(FAKE_FAILURE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _preview_text(value: Any, *, width: int = 60) -> str:
    if not isinstance(value, str):
        return ""

    collapsed = " ".join(value.split())
    if len(collapsed) <= width:
        return collapsed

    return f"{collapsed[: max(width - 3, 0)].rstrip()}..."
