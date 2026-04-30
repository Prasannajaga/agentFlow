from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from agentflow.services.secret_resolution import SecretResolutionError, parse_secret_ref

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

    @model_validator(mode="after")
    def validate_provider_requirements(self) -> "ProviderConfig":
        if self.type == "fake":
            return self

        if self.type == "openai_compatible":
            if self.base_url is None:
                raise ValueError("provider.base_url is required for openai_compatible.")
            if self.api_key_ref is None:
                raise ValueError("provider.api_key_ref is required for openai_compatible.")
            try:
                parse_secret_ref(self.api_key_ref)
            except SecretResolutionError as exc:
                raise ValueError(str(exc)) from exc
            return self

        raise ValueError(f"Unsupported provider type: {self.type}")


class ExternalCliRunnerConfig(StrictModel):
    type: Literal["external_cli"]
    command: NonEmptyString
    args: list[NonEmptyString] = Field(default_factory=list)
    cwd: NonEmptyString = "."
    timeout_seconds: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_runner_requirements(self) -> "ExternalCliRunnerConfig":
        normalized_cwd = self.cwd.strip()
        cwd_path = PurePosixPath(normalized_cwd)
        if cwd_path.is_absolute():
            raise ValueError("runner.cwd must be a relative path.")
        if ".." in cwd_path.parts:
            raise ValueError("runner.cwd cannot contain parent path traversal ('..').")

        return self


class CursorSdkRunnerConfig(StrictModel):
    type: Literal["cursor_sdk"]
    runtime: Literal["local", "cloud"] = "local"
    model: NonEmptyString = "auto"
    cwd: NonEmptyString = "."
    api_key_ref: NonEmptyString
    timeout_seconds: PositiveInt = 600

    @model_validator(mode="after")
    def validate_runner_requirements(self) -> "CursorSdkRunnerConfig":
        if self.runtime == "cloud":
            raise ValueError("runner.runtime=cloud is not implemented yet. Use runner.runtime=local.")

        normalized_cwd = self.cwd.strip()
        cwd_path = PurePosixPath(normalized_cwd)
        if cwd_path.is_absolute():
            raise ValueError("runner.cwd must be a relative path.")
        if ".." in cwd_path.parts:
            raise ValueError("runner.cwd cannot contain parent path traversal ('..').")

        try:
            parse_secret_ref(self.api_key_ref)
        except SecretResolutionError as exc:
            raise ValueError(str(exc)) from exc

        return self


RunnerConfig = Annotated[
    ExternalCliRunnerConfig | CursorSdkRunnerConfig,
    Field(discriminator="type"),
]


class RetryConfig(StrictModel):
    max_attempts: PositiveInt = 1
    backoff_seconds: NonNegativeInt = 0


class RuntimeConfig(StrictModel):
    max_steps: PositiveInt = 10
    timeout_seconds: PositiveInt = 600
    provider_timeout_seconds: PositiveInt | None = None
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
    provider: ProviderConfig | None = None
    runner: RunnerConfig | None = None
    system_prompt: NonEmptyString
    runtime: RuntimeConfig | None = None
    tools: list[NonEmptyString] = Field(default_factory=list)
    task: TaskConfig | None = None
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)

    @model_validator(mode="after")
    def validate_execution_mode(self) -> "AgentConfig":
        if self.provider is None and self.runner is None:
            raise ValueError("Either provider or runner must be configured.")
        return self
