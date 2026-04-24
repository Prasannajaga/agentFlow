from __future__ import annotations

import json
from typing import Any

from agentflow.tools.base import ToolExecutionError, ToolInvocationRequest, ToolResult

ECHO_TOOL_NAME = "echo"
DEFAULT_ECHO_TEXT = "default echo payload"


class EchoToolAdapter:
    tool_name = ECHO_TOOL_NAME

    def invoke(self, request: ToolInvocationRequest) -> ToolResult:
        try:
            text_value = _normalize_text(request.input_data.get("text"))
        except Exception as exc:
            raise ToolExecutionError(
                self.tool_name,
                f"Echo tool failed to normalize input: {exc}",
                error_type="invalid_input",
            ) from exc

        return ToolResult(
            tool_name=self.tool_name,
            ok=True,
            result={"text": text_value},
        )


def build_echo_input(input_json: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(input_json, dict) and "text" in input_json:
        return {"text": _normalize_text(input_json.get("text"))}
    return {"text": DEFAULT_ECHO_TEXT}


def preview_echo_result(tool_result: ToolResult, *, width: int = 120) -> str:
    text_value = tool_result.result.get("text")
    if not isinstance(text_value, str):
        return ""

    collapsed = " ".join(text_value.split())
    if len(collapsed) <= width:
        return collapsed
    return f"{collapsed[: max(width - 3, 0)].rstrip()}..."


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or DEFAULT_ECHO_TEXT

    if value is None:
        return DEFAULT_ECHO_TEXT

    if isinstance(value, (int, float, bool)):
        return str(value)

    return json.dumps(value, sort_keys=True)
