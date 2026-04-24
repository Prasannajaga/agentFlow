from __future__ import annotations

import os
from typing import Any

from agentflow.providers.base import preview_text
from agentflow.providers.fake import (
    FAKE_FAILURE_ENV,
    FAKE_MODEL_NAME,
    FAKE_PROVIDER_NAME,
    FakeProviderAdapter,
)
from agentflow.services.runtime_validation import build_provider_request


def execute_fake_agent(config: dict[str, Any]) -> dict[str, Any]:
    request = build_provider_request(config)
    return FakeProviderAdapter().invoke(request).to_output_json()


def _should_fail() -> bool:
    return os.environ.get(FAKE_FAILURE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _preview_text(value: Any, *, width: int = 60) -> str:
    if not isinstance(value, str):
        return ""
    return preview_text(value, width=width)
