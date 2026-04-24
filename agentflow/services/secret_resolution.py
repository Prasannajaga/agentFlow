from __future__ import annotations

import os
import re
from collections.abc import Mapping

ENV_SECRET_PREFIX = "env:"
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SecretResolutionError(RuntimeError):
    def __init__(self, message: str, *, error_type: str) -> None:
        super().__init__(message)
        self.error_type = error_type


def parse_secret_ref(secret_ref: str) -> tuple[str, str]:
    if not isinstance(secret_ref, str):
        raise SecretResolutionError(
            "Secret reference must be a string in the format 'env:NAME'.",
            error_type="invalid_secret_ref",
        )

    normalized = secret_ref.strip()
    if not normalized:
        raise SecretResolutionError(
            "Secret reference must be a non-empty string in the format 'env:NAME'.",
            error_type="invalid_secret_ref",
        )

    if not normalized.startswith(ENV_SECRET_PREFIX):
        raise SecretResolutionError(
            "Unsupported secret reference format. Expected 'env:NAME'.",
            error_type="invalid_secret_ref",
        )

    env_var_name = normalized[len(ENV_SECRET_PREFIX) :].strip()
    if not env_var_name:
        raise SecretResolutionError(
            "Secret reference must include an environment variable name after 'env:'.",
            error_type="invalid_secret_ref",
        )

    if _ENV_NAME_PATTERN.fullmatch(env_var_name) is None:
        raise SecretResolutionError(
            f"Invalid environment variable name in secret reference: {env_var_name}",
            error_type="invalid_secret_ref",
        )

    return ("env", env_var_name)


def resolve_secret_ref(secret_ref: str, *, env: Mapping[str, str] | None = None) -> str:
    source, key = parse_secret_ref(secret_ref)
    if source != "env":  # pragma: no cover - defensive guard
        raise SecretResolutionError(
            "Unsupported secret source.",
            error_type="invalid_secret_ref",
        )

    values = env if env is not None else os.environ
    value = values.get(key)
    if isinstance(value, str) and value.strip():
        return value

    raise SecretResolutionError(
        f"Environment variable {key} is not set.",
        error_type="missing_secret",
    )
