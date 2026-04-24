from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ToolError(RuntimeError):
    def __init__(
        self,
        tool_name: str,
        message: str,
        *,
        error_type: str,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.error_type = error_type


class ToolConfigurationError(ToolError):
    def __init__(self, tool_name: str, message: str, *, error_type: str = "invalid_tool_config") -> None:
        super().__init__(tool_name, message, error_type=error_type)


class ToolExecutionError(ToolError):
    def __init__(self, tool_name: str, message: str, *, error_type: str = "tool_execution_failed") -> None:
        super().__init__(tool_name, message, error_type=error_type)


@dataclass(frozen=True)
class ToolInvocationRequest:
    tool_name: str
    input_data: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    ok: bool
    result: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "ok": self.ok,
            "result": self.result,
        }


class ToolAdapter(Protocol):
    tool_name: str

    def invoke(self, request: ToolInvocationRequest) -> ToolResult:
        ...
