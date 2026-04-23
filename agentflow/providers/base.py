from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_FALLBACK_USER_MESSAGE = "Execute the configured agent task."


class ProviderError(RuntimeError):
    def __init__(
        self,
        provider_type: str,
        message: str,
        *,
        error_type: str,
    ) -> None:
        super().__init__(message)
        self.provider_type = provider_type
        self.error_type = error_type


class ProviderConfigurationError(ProviderError):
    def __init__(self, provider_type: str, message: str, *, error_type: str = "invalid_provider_config") -> None:
        super().__init__(provider_type, message, error_type=error_type)


class ProviderExecutionError(ProviderError):
    def __init__(self, provider_type: str, message: str, *, error_type: str = "provider_execution_failed") -> None:
        super().__init__(provider_type, message, error_type=error_type)


@dataclass(frozen=True)
class ProviderInvocationRequest:
    provider_type: str
    model: str | None
    base_url: str | None
    api_key_ref: str | None
    system_prompt: str
    input_json: dict[str, Any] | None
    resolved_config_json: dict[str, Any]
    timeout_seconds: int

    @classmethod
    def from_resolved_config(
        cls,
        resolved_config_json: dict[str, Any],
        *,
        input_json: dict[str, Any] | None = None,
    ) -> "ProviderInvocationRequest":
        provider_config = resolved_config_json.get("provider")
        if not isinstance(provider_config, dict):
            raise ProviderConfigurationError("unknown", "Resolved config is missing a provider object.")

        provider_type = str(provider_config.get("type") or "").strip()
        if not provider_type:
            raise ProviderConfigurationError("unknown", "Resolved config is missing provider.type.")

        runtime_config = resolved_config_json.get("runtime")
        timeout_seconds = 60
        if isinstance(runtime_config, dict):
            configured_timeout = runtime_config.get("timeout_seconds")
            if isinstance(configured_timeout, int) and configured_timeout > 0:
                timeout_seconds = configured_timeout

        return cls(
            provider_type=provider_type,
            model=_coerce_optional_string(provider_config.get("model")),
            base_url=_coerce_optional_string(provider_config.get("base_url")),
            api_key_ref=_coerce_optional_string(provider_config.get("api_key_ref")),
            system_prompt=_coerce_optional_string(resolved_config_json.get("system_prompt")) or "",
            input_json=dict(input_json) if input_json is not None else None,
            resolved_config_json=dict(resolved_config_json),
            timeout_seconds=timeout_seconds,
        )

    def build_messages(self) -> list[dict[str, str]]:
        user_message = DEFAULT_FALLBACK_USER_MESSAGE
        if self.input_json:
            user_message = f"Run input:\n{json.dumps(self.input_json, indent=2, sort_keys=True)}"

        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_message})
        return messages


@dataclass(frozen=True)
class ProviderResult:
    provider_type: str
    model: str
    output_text: str
    finish_reason: str | None = None
    raw_response: dict[str, Any] | None = None

    def to_output_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider_type": self.provider_type,
            "model": self.model,
            "output_text": self.output_text,
        }
        if self.finish_reason is not None:
            payload["finish_reason"] = self.finish_reason
        if self.raw_response is not None:
            payload["raw_response"] = self.raw_response
        return payload


class ProviderAdapter(Protocol):
    provider_type: str

    def describe_request(self, request: ProviderInvocationRequest) -> dict[str, Any]:
        ...

    def invoke(self, request: ProviderInvocationRequest) -> ProviderResult:
        ...


def preview_text(value: str, *, width: int = 120) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= width:
        return collapsed
    return f"{collapsed[: max(width - 3, 0)].rstrip()}..."


def _coerce_optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None
