from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentflow.schemas.agent_config import AgentConfig


class AgentYamlError(ValueError):
    """Raised when an agent YAML file cannot be parsed into a config payload."""


def format_yaml_error(error: yaml.YAMLError) -> str:
    mark = getattr(error, "problem_mark", None)
    if mark is None:
        return str(error)

    problem = getattr(error, "problem", None) or "Invalid YAML syntax"
    return f"{problem} at line {mark.line + 1}, column {mark.column + 1}"


def load_yaml_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        try:
            payload = yaml.safe_load(handle)
        except yaml.YAMLError as exc:  # pragma: no cover - depends on parser detail
            raise AgentYamlError(f"YAML parsing error: {format_yaml_error(exc)}") from exc

    if payload is None:
        raise AgentYamlError("YAML file is empty.")

    if not isinstance(payload, dict):
        raise AgentYamlError("Top-level YAML value must be an object/mapping.")

    return payload


def load_agent_config(path: Path) -> AgentConfig:
    payload = load_yaml_payload(path)
    return AgentConfig.model_validate(payload)
