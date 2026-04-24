from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentflow.providers.base import (
    ProviderConfigurationError,
    ProviderError,
    ProviderExecutionError,
    ProviderInvocationRequest,
)
from agentflow.providers.openai_compatible import OPENAI_COMPATIBLE_PROVIDER_TYPE
from agentflow.providers.registry import get_provider_adapter
from agentflow.services.secret_resolution import SecretResolutionError, parse_secret_ref, resolve_secret_ref
from agentflow.tools.base import ToolConfigurationError, ToolError, ToolExecutionError
from agentflow.tools.registry import validate_tool_names

RUNTIME_ERROR_CONFIG = "config_error"
RUNTIME_ERROR_SECRET = "secret_error"
RUNTIME_ERROR_PROVIDER_SETUP = "provider_setup_error"
RUNTIME_ERROR_TOOL_VALIDATION = "tool_validation_error"
RUNTIME_ERROR_PROVIDER_EXECUTION = "provider_execution_error"

SUPPORTED_PROVIDER_TYPES = frozenset({"fake", OPENAI_COMPATIBLE_PROVIDER_TYPE})
DEFAULT_RUNTIME_TIMEOUT_SECONDS = 600
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 60


class RuntimeValidationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        classification: str,
        error_type: str,
    ) -> None:
        super().__init__(message)
        self.classification = classification
        self.error_type = error_type


@dataclass(frozen=True)
class ValidatedProviderConfig:
    provider_type: str
    model: str
    base_url: str | None
    api_key_ref: str | None


@dataclass(frozen=True)
class ResolvedRuntimeTimeouts:
    timeout_seconds: int
    provider_timeout_seconds: int


@dataclass(frozen=True)
class ValidatedRunConfiguration:
    provider: ValidatedProviderConfig
    tools: tuple[str, ...]
    timeouts: ResolvedRuntimeTimeouts


def validate_provider_config(resolved_config_json: dict[str, Any]) -> ValidatedProviderConfig:
    provider_config = resolved_config_json.get("provider")
    if not isinstance(provider_config, dict):
        raise RuntimeValidationError(
            "Resolved config is missing a provider object.",
            classification=RUNTIME_ERROR_CONFIG,
            error_type="invalid_provider_config",
        )

    provider_type = _require_non_empty_string(
        provider_config.get("type"),
        "provider.type is required.",
    )
    if provider_type not in SUPPORTED_PROVIDER_TYPES:
        raise RuntimeValidationError(
            f"Unsupported provider type: {provider_type}",
            classification=RUNTIME_ERROR_CONFIG,
            error_type="unsupported_provider",
        )

    model = _require_non_empty_string(
        provider_config.get("model"),
        f"provider.model is required for {provider_type}",
    )

    base_url = _optional_non_empty_string(provider_config.get("base_url"))
    api_key_ref = _optional_non_empty_string(provider_config.get("api_key_ref"))

    if provider_type == OPENAI_COMPATIBLE_PROVIDER_TYPE:
        if base_url is None:
            raise RuntimeValidationError(
                "provider.base_url is required for openai_compatible",
                classification=RUNTIME_ERROR_CONFIG,
                error_type="invalid_provider_config",
            )
        if api_key_ref is None:
            raise RuntimeValidationError(
                "provider.api_key_ref is required for openai_compatible",
                classification=RUNTIME_ERROR_CONFIG,
                error_type="invalid_provider_config",
            )

        try:
            parse_secret_ref(api_key_ref)
        except SecretResolutionError as exc:
            raise RuntimeValidationError(
                str(exc),
                classification=RUNTIME_ERROR_SECRET,
                error_type=exc.error_type,
            ) from exc

    return ValidatedProviderConfig(
        provider_type=provider_type,
        model=model,
        base_url=base_url,
        api_key_ref=api_key_ref,
    )


def validate_tool_config(resolved_config_json: dict[str, Any]) -> tuple[str, ...]:
    tools_value = resolved_config_json.get("tools")
    if tools_value is None:
        return ()

    if not isinstance(tools_value, list):
        raise RuntimeValidationError(
            "tools must be a list of tool names.",
            classification=RUNTIME_ERROR_TOOL_VALIDATION,
            error_type="invalid_tool_config",
        )

    normalized_input: list[str] = []
    for index, value in enumerate(tools_value):
        if not isinstance(value, str):
            raise RuntimeValidationError(
                f"tools[{index}] must be a string tool name.",
                classification=RUNTIME_ERROR_TOOL_VALIDATION,
                error_type="invalid_tool_name",
            )
        normalized_input.append(value)

    try:
        return validate_tool_names(normalized_input)
    except ToolConfigurationError as exc:
        raise RuntimeValidationError(
            str(exc),
            classification=RUNTIME_ERROR_TOOL_VALIDATION,
            error_type=exc.error_type,
        ) from exc


def resolve_provider_timeout(
    resolved_config_json: dict[str, Any],
    *,
    default_runtime_timeout_seconds: int = DEFAULT_RUNTIME_TIMEOUT_SECONDS,
    default_provider_timeout_seconds: int = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
) -> ResolvedRuntimeTimeouts:
    runtime_value = resolved_config_json.get("runtime")
    if runtime_value is None:
        return ResolvedRuntimeTimeouts(
            timeout_seconds=default_runtime_timeout_seconds,
            provider_timeout_seconds=default_provider_timeout_seconds,
        )

    if not isinstance(runtime_value, dict):
        raise RuntimeValidationError(
            "runtime must be an object when provided.",
            classification=RUNTIME_ERROR_CONFIG,
            error_type="invalid_runtime_config",
        )

    timeout_seconds = default_runtime_timeout_seconds
    if "timeout_seconds" in runtime_value:
        timeout_seconds = _require_positive_int(runtime_value.get("timeout_seconds"), "runtime.timeout_seconds")

    provider_timeout_seconds = default_provider_timeout_seconds
    if "provider_timeout_seconds" in runtime_value:
        provider_timeout_seconds = _require_positive_int(
            runtime_value.get("provider_timeout_seconds"),
            "runtime.provider_timeout_seconds",
        )

    return ResolvedRuntimeTimeouts(
        timeout_seconds=timeout_seconds,
        provider_timeout_seconds=provider_timeout_seconds,
    )


def validate_run_configuration(resolved_config_json: dict[str, Any]) -> ValidatedRunConfiguration:
    provider = validate_provider_config(resolved_config_json)
    tools = validate_tool_config(resolved_config_json)
    timeouts = resolve_provider_timeout(resolved_config_json)
    return ValidatedRunConfiguration(provider=provider, tools=tools, timeouts=timeouts)


def build_provider_request(
    resolved_config_json: dict[str, Any],
    *,
    input_json: dict[str, Any] | None = None,
) -> ProviderInvocationRequest:
    validated = validate_run_configuration(resolved_config_json)
    return ProviderInvocationRequest(
        provider_type=validated.provider.provider_type,
        model=validated.provider.model,
        base_url=validated.provider.base_url,
        api_key_ref=validated.provider.api_key_ref,
        system_prompt=_optional_non_empty_string(resolved_config_json.get("system_prompt")) or "",
        input_json=dict(input_json) if input_json is not None else None,
        resolved_config_json=dict(resolved_config_json),
        timeout_seconds=validated.timeouts.provider_timeout_seconds,
    )


def validate_provider_setup(request: ProviderInvocationRequest) -> None:
    try:
        get_provider_adapter(request)
    except ProviderConfigurationError as exc:
        raise RuntimeValidationError(
            str(exc),
            classification=RUNTIME_ERROR_PROVIDER_SETUP,
            error_type=exc.error_type,
        ) from exc

    if request.provider_type == OPENAI_COMPATIBLE_PROVIDER_TYPE:
        try:
            resolve_secret_ref(request.api_key_ref or "")
        except SecretResolutionError as exc:
            raise RuntimeValidationError(
                str(exc),
                classification=RUNTIME_ERROR_SECRET,
                error_type=exc.error_type,
            ) from exc


def classify_runtime_error(error: Exception) -> str:
    if isinstance(error, RuntimeValidationError):
        return error.classification

    if isinstance(error, SecretResolutionError):
        return RUNTIME_ERROR_SECRET

    if isinstance(error, ToolConfigurationError):
        return RUNTIME_ERROR_TOOL_VALIDATION

    if isinstance(error, ProviderConfigurationError):
        return _classify_provider_configuration_error(error)

    if isinstance(error, (ProviderExecutionError, ToolExecutionError)):
        return RUNTIME_ERROR_PROVIDER_EXECUTION

    if isinstance(error, ToolError):
        return RUNTIME_ERROR_PROVIDER_EXECUTION

    if isinstance(error, ProviderError):
        return RUNTIME_ERROR_PROVIDER_SETUP

    return RUNTIME_ERROR_PROVIDER_EXECUTION


def _classify_provider_configuration_error(error: ProviderConfigurationError) -> str:
    if error.error_type in {
        RUNTIME_ERROR_CONFIG,
        RUNTIME_ERROR_SECRET,
        RUNTIME_ERROR_PROVIDER_SETUP,
        RUNTIME_ERROR_TOOL_VALIDATION,
        RUNTIME_ERROR_PROVIDER_EXECUTION,
    }:
        return error.error_type

    if error.error_type in {"invalid_secret_ref", "missing_secret"}:
        return RUNTIME_ERROR_SECRET

    if error.error_type in {"unsupported_provider", "invalid_provider_config", "invalid_runtime_config"}:
        return RUNTIME_ERROR_CONFIG

    return RUNTIME_ERROR_PROVIDER_SETUP


def _require_non_empty_string(value: Any, message: str) -> str:
    normalized = _optional_non_empty_string(value)
    if normalized is None:
        raise RuntimeValidationError(
            message,
            classification=RUNTIME_ERROR_CONFIG,
            error_type="invalid_provider_config",
        )
    return normalized


def _optional_non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise RuntimeValidationError(
            f"{field_name} must be a positive integer.",
            classification=RUNTIME_ERROR_CONFIG,
            error_type="invalid_runtime_config",
        )
    return value
