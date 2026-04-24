from __future__ import annotations

from typing import Any

DEFAULT_MAX_ATTEMPTS = 1


def extract_max_attempts(resolved_config_json: dict[str, Any]) -> int:
    runtime_config = resolved_config_json.get("runtime")
    if not isinstance(runtime_config, dict):
        return DEFAULT_MAX_ATTEMPTS

    retry_config = runtime_config.get("retry")
    if not isinstance(retry_config, dict):
        return DEFAULT_MAX_ATTEMPTS

    max_attempts = retry_config.get("max_attempts")
    if isinstance(max_attempts, int) and max_attempts >= 1:
        return max_attempts

    return DEFAULT_MAX_ATTEMPTS


def should_retry_failed_attempt(
    *,
    attempt_count: int,
    max_attempts: int,
    retryable: bool,
) -> bool:
    return retryable and attempt_count < max_attempts
