from __future__ import annotations

import pytest

from agentflow.providers.fake import FAKE_FAILURE_ENV
from agentflow.services.agent_registry import register_agent_from_yaml_text
from agentflow.services.agent_runner import (
    AgentRunExecutionFailedError,
    create_run_for_agent,
    execute_claimed_run,
)
from agentflow.services.run_events import (
    RUN_EVENT_PROVIDER_EXECUTION_COMPLETED,
    RUN_EVENT_RUN_COMPLETED,
    list_run_events,
)
from agentflow.services.run_queries import (
    RUN_STATUS_FAILED,
    RUN_STATUS_PENDING,
    RUN_STATUS_RUNNING,
    claim_next_pending_run,
    get_agent_run,
)
from conftest import make_agent_yaml

pytestmark = pytest.mark.db


def test_create_run_persists_pending_snapshot_and_attempt_defaults(db_session_factory) -> None:
    registration = register_agent_from_yaml_text(
        make_agent_yaml(extra={"runtime": {"retry": {"max_attempts": 3}}}),
        session_factory=db_session_factory,
    )

    prepared = create_run_for_agent(
        registration.agent_id,
        input_json={"topic": "deterministic test"},
        session_factory=db_session_factory,
    )
    run = prepared.run

    assert run.status == RUN_STATUS_PENDING
    assert run.input_json == {"topic": "deterministic test"}
    assert run.resolved_config_json["name"] == "test-agent"
    assert run.resolved_config_json["provider"]["type"] == "fake"
    assert run.attempt_count == 0
    assert run.max_attempts == 3


def test_updating_agent_does_not_mutate_existing_run_snapshot(db_session_factory) -> None:
    first = register_agent_from_yaml_text(
        make_agent_yaml(name="snapshot-agent", extra={"system_prompt": "snapshot-v1"}),
        session_factory=db_session_factory,
    )

    prepared = create_run_for_agent(first.agent_id, session_factory=db_session_factory)

    second = register_agent_from_yaml_text(
        make_agent_yaml(name="snapshot-agent", extra={"system_prompt": "snapshot-v2"}),
        agent_id=first.agent_id,
        session_factory=db_session_factory,
    )

    stored_run = get_agent_run(prepared.run.run_id, session_factory=db_session_factory)

    assert stored_run is not None
    assert stored_run.version_id == first.version_id
    assert stored_run.version_id != second.version_id
    assert stored_run.resolved_config_json["system_prompt"] == "snapshot-v1"


def test_fake_provider_execution_completes_run_and_records_events(db_session_factory) -> None:
    registration = register_agent_from_yaml_text(make_agent_yaml(), session_factory=db_session_factory)
    prepared = create_run_for_agent(registration.agent_id, session_factory=db_session_factory)

    claimed = claim_next_pending_run(worker_id="worker-test", session_factory=db_session_factory)
    assert claimed is not None
    assert claimed.run_id == prepared.run.run_id
    assert claimed.status == RUN_STATUS_RUNNING

    completed = execute_claimed_run(claimed, session_factory=db_session_factory)

    assert completed.status == "completed"
    assert completed.output_json is not None
    assert completed.output_json["output_text"] == "Fake run completed successfully."

    events = list_run_events(completed.run_id, session_factory=db_session_factory)
    event_types = {event.event_type for event in events}
    assert RUN_EVENT_PROVIDER_EXECUTION_COMPLETED in event_types
    assert RUN_EVENT_RUN_COMPLETED in event_types


def test_failing_run_retries_then_marks_failed(monkeypatch: pytest.MonkeyPatch, db_session_factory) -> None:
    monkeypatch.setenv(FAKE_FAILURE_ENV, "1")

    registration = register_agent_from_yaml_text(
        make_agent_yaml(extra={"runtime": {"retry": {"max_attempts": 2}}}),
        session_factory=db_session_factory,
    )
    prepared = create_run_for_agent(registration.agent_id, session_factory=db_session_factory)

    first_claim = claim_next_pending_run(worker_id="worker-test", session_factory=db_session_factory)
    assert first_claim is not None
    assert first_claim.run_id == prepared.run.run_id

    with pytest.raises(AgentRunExecutionFailedError) as first_error:
        execute_claimed_run(first_claim, session_factory=db_session_factory)

    first_result = first_error.value.run
    assert first_result.status == RUN_STATUS_PENDING
    assert first_result.attempt_count == 1
    assert first_result.max_attempts == 2

    second_claim = claim_next_pending_run(worker_id="worker-test", session_factory=db_session_factory)
    assert second_claim is not None

    with pytest.raises(AgentRunExecutionFailedError) as second_error:
        execute_claimed_run(second_claim, session_factory=db_session_factory)

    second_result = second_error.value.run
    assert second_result.status == RUN_STATUS_FAILED
    assert second_result.attempt_count == 2
    assert second_result.max_attempts == 2
