from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
LogLevel = Literal["debug", "info", "warning", "error"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class ProviderConfig(StrictModel):
    type: NonEmptyString
    model: NonEmptyString
    base_url: NonEmptyString | None = None
    api_key_ref: NonEmptyString | None = None


class RetryConfig(StrictModel):
    max_attempts: PositiveInt = 1
    backoff_seconds: NonNegativeInt = 0


class RuntimeConfig(StrictModel):
    max_steps: PositiveInt = 10
    timeout_seconds: PositiveInt = 600
    retry: RetryConfig = Field(default_factory=RetryConfig)


class TaskConfig(StrictModel):
    type: NonEmptyString | None = None
    input_schema: dict[NonEmptyString, NonEmptyString] = Field(default_factory=dict)


class TelemetryConfig(StrictModel):
    trace_enabled: bool = False
    log_level: LogLevel = "info"
    store_prompts: bool = False


class OutputsConfig(StrictModel):
    save_final_text: bool = False
    save_artifacts: bool = False


class AgentConfig(StrictModel):
    name: NonEmptyString
    version: PositiveInt
    description: NonEmptyString | None = None
    provider: ProviderConfig
    system_prompt: NonEmptyString
    runtime: RuntimeConfig | None = None
    tools: list[NonEmptyString] = Field(default_factory=list)
    task: TaskConfig | None = None
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
