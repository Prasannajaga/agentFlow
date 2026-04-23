from __future__ import annotations

import os
from typing import Any

from agentflow.providers.base import ProviderInvocationRequest, preview_text
from agentflow.providers.fake import (
    FAKE_FAILURE_ENV,
    FAKE_MODEL_NAME,
    FAKE_PROVIDER_NAME,
    FakeProviderAdapter,
)


def execute_fake_agent(config: dict[str, Any]) -> dict[str, Any]:
    request = ProviderInvocationRequest.from_resolved_config(config)
    return FakeProviderAdapter().invoke(request).to_output_json()


def _should_fail() -> bool:
    return os.environ.get(FAKE_FAILURE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _preview_text(value: Any, *, width: int = 60) -> str:
    if not isinstance(value, str):
        return ""
    return preview_text(value, width=width)
