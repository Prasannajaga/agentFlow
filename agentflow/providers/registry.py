from __future__ import annotations

from agentflow.providers.base import ProviderConfigurationError, ProviderInvocationRequest
from agentflow.providers.fake import FAKE_PROVIDER_NAME, FakeProviderAdapter
from agentflow.providers.openai_compatible import (
    OPENAI_COMPATIBLE_PROVIDER_TYPE,
    OpenAICompatibleProviderAdapter,
)

_PROVIDERS = {
    FAKE_PROVIDER_NAME: FakeProviderAdapter(),
    OPENAI_COMPATIBLE_PROVIDER_TYPE: OpenAICompatibleProviderAdapter(),
}


def get_provider_adapter(request: ProviderInvocationRequest):
    adapter = _PROVIDERS.get(request.provider_type)
    if adapter is not None:
        return adapter

    raise ProviderConfigurationError(
        request.provider_type,
        f"Unsupported provider type: {request.provider_type}",
        error_type="unsupported_provider",
    )
