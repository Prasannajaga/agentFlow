from __future__ import annotations

import pytest

from agentflow.evaluators.exact_match import EVALUATOR_TYPE
from agentflow.services.agent_registry import register_agent_from_yaml_text
from agentflow.services.agent_runner import create_run_for_agent, execute_claimed_run
from agentflow.services.eval_service import evaluate_run, list_run_evaluations
from agentflow.services.run_queries import claim_next_pending_run, mark_agent_run_completed
from conftest import make_agent_yaml

pytestmark = pytest.mark.db


def test_exact_match_passes_for_matching_text(db_session_factory) -> None:
    completed_run = _create_completed_run(db_session_factory)

    evaluation = evaluate_run(
        completed_run.run_id,
        evaluator_type=EVALUATOR_TYPE,
        expected_text="Fake run completed successfully.",
        session_factory=db_session_factory,
    )

    assert evaluation.status == "completed"
    assert evaluation.passed is True
    assert evaluation.score == 1.0


def test_exact_match_fails_for_different_text(db_session_factory) -> None:
    completed_run = _create_completed_run(db_session_factory)

    evaluation = evaluate_run(
        completed_run.run_id,
        evaluator_type=EVALUATOR_TYPE,
        expected_text="not the same",
        session_factory=db_session_factory,
    )

    assert evaluation.status == "completed"
    assert evaluation.passed is False
    assert evaluation.score == 0.0


def test_evaluation_record_persists_score_and_pass_fail(db_session_factory) -> None:
    completed_run = _create_completed_run(db_session_factory)

    evaluation = evaluate_run(
        completed_run.run_id,
        evaluator_type=EVALUATOR_TYPE,
        expected_text="not the same",
        session_factory=db_session_factory,
    )

    persisted = list_run_evaluations(completed_run.run_id, session_factory=db_session_factory)

    assert len(persisted) == 1
    assert persisted[0].evaluation_id == evaluation.evaluation_id
    assert persisted[0].score == 0.0
    assert persisted[0].passed is False


def test_evaluator_handles_missing_output_text_cleanly(db_session_factory) -> None:
    registration = register_agent_from_yaml_text(make_agent_yaml(), session_factory=db_session_factory)
    prepared = create_run_for_agent(registration.agent_id, session_factory=db_session_factory)

    completed = mark_agent_run_completed(
        prepared.run.run_id,
        output_json={},
        session_factory=db_session_factory,
    )
    assert completed is not None

    evaluation = evaluate_run(
        completed.run_id,
        evaluator_type=EVALUATOR_TYPE,
        expected_text="",
        session_factory=db_session_factory,
    )

    assert evaluation.passed is True
    assert evaluation.actual_json == {"text": ""}


def _create_completed_run(db_session_factory):
    registration = register_agent_from_yaml_text(make_agent_yaml(), session_factory=db_session_factory)
    prepared = create_run_for_agent(registration.agent_id, session_factory=db_session_factory)
    claimed = claim_next_pending_run(worker_id="eval-worker", session_factory=db_session_factory)
    assert claimed is not None
    return execute_claimed_run(claimed, session_factory=db_session_factory)
