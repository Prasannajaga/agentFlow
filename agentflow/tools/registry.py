from __future__ import annotations

from collections.abc import Sequence

from agentflow.tools.base import ToolConfigurationError
from agentflow.tools.echo import ECHO_TOOL_NAME, EchoToolAdapter

_TOOLS = {
    ECHO_TOOL_NAME: EchoToolAdapter(),
}


def validate_tool_names(tool_names: Sequence[str]) -> tuple[str, ...]:
    normalized_tools: list[str] = []
    seen: set[str] = set()

    for tool_name in tool_names:
        normalized_name = tool_name.strip()
        if not normalized_name:
            raise ToolConfigurationError("unknown", "Tool names must not be blank.", error_type="invalid_tool_name")
        if normalized_name not in _TOOLS:
            raise ToolConfigurationError(
                normalized_name,
                f"Unsupported tool: {normalized_name}",
                error_type="unsupported_tool",
            )
        if normalized_name in seen:
            raise ToolConfigurationError(
                normalized_name,
                f"Duplicate tool entry is not allowed: {normalized_name}",
                error_type="duplicate_tool",
            )
        seen.add(normalized_name)
        normalized_tools.append(normalized_name)

    return tuple(normalized_tools)


def get_tool_adapter(tool_name: str):
    adapter = _TOOLS.get(tool_name)
    if adapter is not None:
        return adapter

    raise ToolConfigurationError(
        tool_name,
        f"Unsupported tool: {tool_name}",
        error_type="unsupported_tool",
    )
