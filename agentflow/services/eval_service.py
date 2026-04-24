from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from agentflow.db.models import AgentRun, RunBatchItem, RunEvaluation, utc_now
from agentflow.db.session import create_session_factory
from agentflow.evaluators.registry import EvaluatorNotFoundError, get_evaluator
from agentflow.services.run_queries import RUN_STATUS_COMPLETED, AgentRunDetail, get_agent_run

EVALUATION_STATUS_COMPLETED = "completed"


class EvalError(RuntimeError):
    """Base error for evaluation operations."""


class EvalRunNotFoundError(EvalError):
    pass


class EvalBatchNotFoundError(EvalError):
    pass


class EvalRunIneligibleError(EvalError):
    pass


class EvalInvalidError(EvalError):
    pass


@dataclass(frozen=True)
class RunEvaluationRecord:
    evaluation_id: uuid.UUID
    run_id: uuid.UUID
    evaluator_type: str
    status: str
    score: float | None
    passed: bool | None
    summary: str | None
    expected_json: dict[str, object] | None
    actual_json: dict[str, object] | None
    created_at: datetime


@dataclass(frozen=True)
class BatchEvaluationResult:
    batch_id: uuid.UUID
    evaluated_count: int
    passed_count: int
    failed_count: int
    skipped_count: int
    evaluations: list[RunEvaluationRecord]


@dataclass(frozen=True)
class BatchEvaluationSummary:
    evaluated_count: int
    passed_count: int
    failed_count: int
    latest_created_at: datetime | None


def evaluate_run(
    run_id: uuid.UUID,
    *,
    evaluator_type: str,
    expected_text: str,
    session_factory: sessionmaker[Session] | None = None,
) -> RunEvaluationRecord:
    if expected_text is None:
        raise EvalInvalidError("Expected text is required.")

    run = get_agent_run(run_id, session_factory=session_factory)
    if run is None:
        raise EvalRunNotFoundError(f"Run not found: {run_id}")
    if run.status != RUN_STATUS_COMPLETED:
        raise EvalRunIneligibleError(f"Run must be completed before evaluation: {run_id}")

    return _evaluate_completed_run(
        run,
        evaluator_type=evaluator_type,
        expected_text=expected_text,
        session_factory=session_factory,
    )


def evaluate_batch(
    batch_id: uuid.UUID,
    *,
    evaluator_type: str,
    expected_text: str,
    session_factory: sessionmaker[Session] | None = None,
) -> BatchEvaluationResult:
    session_factory = session_factory or create_session_factory()

    with session_factory() as session:
        run_rows = session.execute(
            select(AgentRun.id, AgentRun.status)
            .join(RunBatchItem, RunBatchItem.run_id == AgentRun.id)
            .where(RunBatchItem.batch_id == batch_id)
            .order_by(RunBatchItem.created_at.asc(), RunBatchItem.id.asc())
        ).all()

    if not run_rows:
        raise EvalBatchNotFoundError(f"Batch not found or has no runs: {batch_id}")

    evaluations: list[RunEvaluationRecord] = []
    skipped_count = 0
    for row in run_rows:
        if row.status != RUN_STATUS_COMPLETED:
            skipped_count += 1
            continue
        evaluations.append(
            evaluate_run(
                row.id,
                evaluator_type=evaluator_type,
                expected_text=expected_text,
                session_factory=session_factory,
            )
        )

    return BatchEvaluationResult(
        batch_id=batch_id,
        evaluated_count=len(evaluations),
        passed_count=sum(1 for evaluation in evaluations if evaluation.passed is True),
        failed_count=sum(1 for evaluation in evaluations if evaluation.passed is False),
        skipped_count=skipped_count,
        evaluations=evaluations,
    )


def list_run_evaluations(
    run_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> list[RunEvaluationRecord]:
    session_factory = session_factory or create_session_factory()
    with session_factory() as session:
        rows = session.execute(
            select(RunEvaluation)
            .where(RunEvaluation.run_id == run_id)
            .order_by(RunEvaluation.created_at.desc(), RunEvaluation.id.desc())
        ).scalars().all()
    return [_build_evaluation_record(row) for row in rows]


def list_latest_evaluations_for_runs(
    run_ids: list[uuid.UUID],
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> dict[uuid.UUID, RunEvaluationRecord]:
    if not run_ids:
        return {}
    session_factory = session_factory or create_session_factory()
    latest: dict[uuid.UUID, RunEvaluationRecord] = {}
    with session_factory() as session:
        rows = session.execute(
            select(RunEvaluation)
            .where(RunEvaluation.run_id.in_(run_ids))
            .order_by(RunEvaluation.run_id.asc(), RunEvaluation.created_at.desc(), RunEvaluation.id.desc())
        ).scalars().all()
    for row in rows:
        latest.setdefault(row.run_id, _build_evaluation_record(row))
    return latest


def get_batch_evaluation_summary(
    batch_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> BatchEvaluationSummary:
    session_factory = session_factory or create_session_factory()
    with session_factory() as session:
        rows = session.execute(
            select(RunEvaluation.passed, RunEvaluation.created_at)
            .join(RunBatchItem, RunBatchItem.run_id == RunEvaluation.run_id)
            .where(RunBatchItem.batch_id == batch_id)
        ).all()
    return BatchEvaluationSummary(
        evaluated_count=len(rows),
        passed_count=sum(1 for row in rows if row.passed is True),
        failed_count=sum(1 for row in rows if row.passed is False),
        latest_created_at=max((row.created_at for row in rows), default=None),
    )


def _evaluate_completed_run(
    run: AgentRunDetail,
    *,
    evaluator_type: str,
    expected_text: str,
    session_factory: sessionmaker[Session] | None,
) -> RunEvaluationRecord:
    try:
        evaluator = get_evaluator(evaluator_type)
    except EvaluatorNotFoundError as exc:
        raise EvalInvalidError(str(exc)) from exc

    result = evaluator.evaluate(run, expected_text=expected_text)
    session_factory = session_factory or create_session_factory()
    now = utc_now()
    with session_factory() as session:
        with session.begin():
            row = RunEvaluation(
                run_id=run.run_id,
                evaluator_type=result.evaluator_type,
                status=EVALUATION_STATUS_COMPLETED,
                score=result.score,
                passed=result.passed,
                summary=result.summary,
                expected_json=result.expected_json,
                actual_json=result.actual_json,
                created_at=now,
            )
            session.add(row)
            session.flush()
            return _build_evaluation_record(row)


def _build_evaluation_record(row: RunEvaluation) -> RunEvaluationRecord:
    return RunEvaluationRecord(
        evaluation_id=row.id,
        run_id=row.run_id,
        evaluator_type=row.evaluator_type,
        status=row.status,
        score=row.score,
        passed=row.passed,
        summary=row.summary,
        expected_json=dict(row.expected_json) if row.expected_json is not None else None,
        actual_json=dict(row.actual_json) if row.actual_json is not None else None,
        created_at=row.created_at,
    )
