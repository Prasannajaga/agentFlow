from __future__ import annotations

from agentflow.evaluators.base import Evaluator
from agentflow.evaluators.exact_match import EVALUATOR_TYPE, ExactMatchEvaluator


class EvaluatorNotFoundError(ValueError):
    pass


def get_evaluator(evaluator_type: str) -> Evaluator:
    if evaluator_type == EVALUATOR_TYPE:
        return ExactMatchEvaluator()
    raise EvaluatorNotFoundError(f"Unknown evaluator: {evaluator_type}")
