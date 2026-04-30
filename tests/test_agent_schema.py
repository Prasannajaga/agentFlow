from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from agentflow.services.yaml_loader import load_agent_document_from_text
from conftest import make_agent_yaml


def test_valid_minimal_fake_provider_yaml_passes() -> None:
    document = load_agent_document_from_text(make_agent_yaml())

    assert document.config.name == "test-agent"
    assert document.config.provider.type == "fake"


def test_unknown_top_level_field_fails_validation() -> None:
    raw_yaml = make_agent_yaml(extra={"unknown_field": "nope"})

    with pytest.raises(ValidationError):
        load_agent_document_from_text(raw_yaml)


def test_missing_required_field_fails_validation() -> None:
    payload = yaml.safe_load(make_agent_yaml())
    payload.pop("system_prompt")

    with pytest.raises(ValidationError):
        load_agent_document_from_text(yaml.safe_dump(payload, sort_keys=False))


def test_invalid_provider_type_fails_validation() -> None:
    raw_yaml = make_agent_yaml(extra={"provider": {"type": "not_real"}})

    with pytest.raises(ValidationError):
        load_agent_document_from_text(raw_yaml)


def test_openai_compatible_provider_missing_fields_fails_validation() -> None:
    raw_yaml = make_agent_yaml(
        extra={
            "provider": {
                "type": "openai_compatible",
                "model": "gpt-4.1-mini",
            }
        }
    )

    with pytest.raises(ValidationError):
        load_agent_document_from_text(raw_yaml)


def test_external_cli_runner_config_is_accepted() -> None:
    raw_yaml = make_agent_yaml(
        extra={
            "provider": None,
            "runner": {
                "type": "external_cli",
                "command": "python",
                "args": ["-c", "print('ok')"],
                "cwd": ".",
                "timeout_seconds": 30,
            },
        }
    )

    document = load_agent_document_from_text(raw_yaml)
    assert document.config.runner is not None
    assert document.config.runner.type == "external_cli"
    assert document.config.runner.command == "python"


def test_external_cli_runner_rejects_invalid_cwd() -> None:
    raw_yaml = make_agent_yaml(
        extra={
            "provider": None,
            "runner": {
                "type": "external_cli",
                "command": "python",
                "args": ["-c", "print('ok')"],
                "cwd": "../outside",
            },
        }
    )

    with pytest.raises(ValidationError):
        load_agent_document_from_text(raw_yaml)


def test_external_cli_runner_rejects_absolute_cwd() -> None:
    raw_yaml = make_agent_yaml(
        extra={
            "provider": None,
            "runner": {
                "type": "external_cli",
                "command": "python",
                "args": ["-c", "print('ok')"],
                "cwd": "/tmp",
            },
        }
    )

    with pytest.raises(ValidationError):
        load_agent_document_from_text(raw_yaml)


def test_external_cli_runner_rejects_non_string_args() -> None:
    raw_yaml = make_agent_yaml(
        extra={
            "provider": None,
            "runner": {
                "type": "external_cli",
                "command": "python",
                "args": ["-c", 123],
                "cwd": ".",
            },
        }
    )

    with pytest.raises(ValidationError):
        load_agent_document_from_text(raw_yaml)


def test_invalid_secret_ref_format_fails_validation() -> None:
    raw_yaml = make_agent_yaml(
        extra={
            "provider": {
                "type": "openai_compatible",
                "model": "gpt-4.1-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key_ref": "OPENAI_API_KEY",
            }
        }
    )

    with pytest.raises(ValidationError):
        load_agent_document_from_text(raw_yaml)


@pytest.mark.parametrize(
    ("runtime_patch",),
    [
        ({"timeout_seconds": 0},),
        ({"retry": {"max_attempts": 0}},),
    ],
)
def test_invalid_timeout_or_retry_values_fail_validation(runtime_patch: dict[str, object]) -> None:
    raw_yaml = make_agent_yaml(extra={"runtime": runtime_patch})

    with pytest.raises(ValidationError):
        load_agent_document_from_text(raw_yaml)
