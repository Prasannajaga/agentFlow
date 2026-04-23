from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agentflow.schemas.agent_config import AgentConfig


class AgentYamlError(ValueError):
    """Raised when an agent YAML file cannot be parsed into a config payload."""


@dataclass(frozen=True)
class AgentDocument:
    raw_yaml: str
    payload: dict[str, Any]
    config: AgentConfig


def format_yaml_error(error: yaml.YAMLError) -> str:
    mark = getattr(error, "problem_mark", None)
    if mark is None:
        return str(error)

    problem = getattr(error, "problem", None) or "Invalid YAML syntax"
    return f"{problem} at line {mark.line + 1}, column {mark.column + 1}"


def read_yaml_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_yaml_text(raw_yaml: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:  # pragma: no cover - depends on parser detail
        raise AgentYamlError(f"YAML parsing error: {format_yaml_error(exc)}") from exc

    if payload is None:
        raise AgentYamlError("YAML file is empty.")

    if not isinstance(payload, dict):
        raise AgentYamlError("Top-level YAML value must be an object/mapping.")

    return payload


def validate_agent_payload(payload: dict[str, Any]) -> AgentConfig:
    return AgentConfig.model_validate(payload)


def normalize_agent_config(config: AgentConfig) -> dict[str, Any]:
    return config.model_dump(mode="json", exclude_none=True)


def load_yaml_payload(path: Path) -> dict[str, Any]:
    raw_yaml = read_yaml_text(path)
    return parse_yaml_text(raw_yaml)


def load_agent_config(path: Path) -> AgentConfig:
    payload = load_yaml_payload(path)
    return validate_agent_payload(payload)


def load_agent_document(path: Path) -> AgentDocument:
    raw_yaml = read_yaml_text(path)
    payload = parse_yaml_text(raw_yaml)
    config = validate_agent_payload(payload)
    return AgentDocument(raw_yaml=raw_yaml, payload=payload, config=config)
