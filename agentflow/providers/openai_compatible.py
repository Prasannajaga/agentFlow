from __future__ import annotations

import json
import os
from typing import Any

import httpx

from agentflow.providers.base import (
    ProviderConfigurationError,
    ProviderExecutionError,
    ProviderInvocationRequest,
    ProviderResult,
)

OPENAI_COMPATIBLE_PROVIDER_TYPE = "openai_compatible"


class OpenAICompatibleProviderAdapter:
    provider_type = OPENAI_COMPATIBLE_PROVIDER_TYPE

    def describe_request(self, request: ProviderInvocationRequest) -> dict[str, Any]:
        return {
            "provider_type": self.provider_type,
            "model": _require_value(request.model, "model"),
            "base_url": _normalize_base_url(_require_value(request.base_url, "base_url")),
        }

    def invoke(self, request: ProviderInvocationRequest) -> ProviderResult:
        model = _require_value(request.model, "model")
        base_url = _normalize_base_url(_require_value(request.base_url, "base_url"))
        api_key_ref = _require_value(request.api_key_ref, "api_key_ref")
        api_key = _resolve_api_key(api_key_ref)
        url = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": request.build_messages(),
        }

        try:
            with httpx.Client(timeout=request.timeout_seconds) as client:
                response = client.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
        except httpx.TimeoutException as exc:
            raise ProviderExecutionError(
                self.provider_type,
                f"Provider request timed out after {request.timeout_seconds} seconds.",
                error_type="timeout",
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderExecutionError(
                self.provider_type,
                f"Provider request failed: {exc}",
                error_type="network_error",
            ) from exc

        if response.status_code >= 400:
            raise ProviderExecutionError(
                self.provider_type,
                f"HTTP {response.status_code}: {_extract_error_message(response)}",
                error_type="http_error",
            )

        try:
            response_json = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderExecutionError(
                self.provider_type,
                "Provider returned invalid JSON.",
                error_type="invalid_response",
            ) from exc

        try:
            choice = response_json["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderExecutionError(
                self.provider_type,
                "Provider response did not include a chat completion choice.",
                error_type="invalid_response",
            ) from exc

        output_text = _extract_output_text(message.get("content"))
        if output_text == "":
            raise ProviderExecutionError(
                self.provider_type,
                "Provider response did not include any output text.",
                error_type="invalid_response",
            )

        finish_reason = choice.get("finish_reason")
        raw_response = {
            "id": response_json.get("id"),
            "object": response_json.get("object"),
            "created": response_json.get("created"),
            "usage": response_json.get("usage"),
            "finish_reason": finish_reason,
        }

        return ProviderResult(
            provider_type=self.provider_type,
            model=model,
            output_text=output_text,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            raw_response=raw_response,
        )


def _require_value(value: str | None, field_name: str) -> str:
    if value:
        return value
    raise ProviderConfigurationError(
        OPENAI_COMPATIBLE_PROVIDER_TYPE,
        f"Provider config is missing '{field_name}'.",
    )


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _resolve_api_key(api_key_ref: str) -> str:
    if not api_key_ref.startswith("env:"):
        raise ProviderConfigurationError(
            OPENAI_COMPATIBLE_PROVIDER_TYPE,
            "Unsupported api_key_ref format. Expected 'env:VARNAME'.",
            error_type="invalid_secret_ref",
        )

    env_var_name = api_key_ref[len("env:") :].strip()
    if not env_var_name:
        raise ProviderConfigurationError(
            OPENAI_COMPATIBLE_PROVIDER_TYPE,
            "api_key_ref must include an environment variable name after 'env:'.",
            error_type="invalid_secret_ref",
        )

    api_key = os.environ.get(env_var_name)
    if api_key:
        return api_key

    raise ProviderConfigurationError(
        OPENAI_COMPATIBLE_PROVIDER_TYPE,
        f"Environment variable '{env_var_name}' is not set.",
        error_type="missing_secret",
    )


def _extract_error_message(response: httpx.Response, *, width: int = 240) -> str:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return _collapse_text(response.text, width=width)

    error_payload = payload.get("error")
    if isinstance(error_payload, dict):
        message = error_payload.get("message")
        if isinstance(message, str) and message.strip():
            return _collapse_text(message, width=width)

    return _collapse_text(json.dumps(payload, sort_keys=True), width=width)


def _extract_output_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
        return "\n".join(part.strip() for part in text_parts if part.strip()).strip()

    return ""


def _collapse_text(value: str, *, width: int) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= width:
        return collapsed
    return f"{collapsed[: max(width - 3, 0)].rstrip()}..."
