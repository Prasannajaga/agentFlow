from __future__ import annotations

import os
from typing import Any

from agentflow.providers.base import ProviderExecutionError, ProviderInvocationRequest, ProviderResult, preview_text

FAKE_PROVIDER_NAME = "fake"
FAKE_MODEL_NAME = "stub-model"
FAKE_FAILURE_ENV = "AGENTFLOW_FAKE_PROVIDER_FAIL"


class FakeProviderAdapter:
    provider_type = FAKE_PROVIDER_NAME

    def describe_request(self, request: ProviderInvocationRequest) -> dict[str, Any]:
        return {
            "provider_type": self.provider_type,
            "model": request.model or FAKE_MODEL_NAME,
        }

    def invoke(self, request: ProviderInvocationRequest) -> ProviderResult:
        if _should_fail():
            raise ProviderExecutionError(
                self.provider_type,
                "Configured fake provider failure.",
                error_type="forced_failure",
            )

        task_config = request.resolved_config_json.get("task")
        task_type = task_config.get("type") if isinstance(task_config, dict) else None

        tools = request.resolved_config_json.get("tools")
        normalized_tools = list(tools) if isinstance(tools, list) else []
        model = request.model or FAKE_MODEL_NAME

        return ProviderResult(
            provider_type=self.provider_type,
            model=model,
            output_text="Fake run completed successfully.",
            finish_reason="stop",
            raw_response={
                "agent_name": request.resolved_config_json.get("name"),
                "system_prompt_preview": preview_text(request.system_prompt, width=60) if request.system_prompt else "",
                "tools": normalized_tools,
                "task_type": task_type,
            },
        )


def _should_fail() -> bool:
    return os.environ.get(FAKE_FAILURE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
