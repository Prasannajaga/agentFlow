from __future__ import annotations

from typing import Any

from agentflow.evaluators.base import EvaluationResult
from agentflow.services.run_queries import AgentRunDetail

EVALUATOR_TYPE = "exact_match"


class ExactMatchEvaluator:
    evaluator_type = EVALUATOR_TYPE

    def evaluate(self, run: AgentRunDetail, *, expected_text: str) -> EvaluationResult:
        actual_text = extract_output_text(run.output_json)
        passed = actual_text == expected_text
        return EvaluationResult(
            evaluator_type=self.evaluator_type,
            passed=passed,
            score=1.0 if passed else 0.0,
            summary="Exact match succeeded." if passed else "Exact match failed.",
            expected_json={"text": expected_text},
            actual_json={"text": actual_text},
        )


def extract_output_text(output_json: dict[str, Any] | None) -> str:
    if not isinstance(output_json, dict):
        return ""

    output_text = output_json.get("output_text")
    if output_text is not None:
        return str(output_text)

    result = output_json.get("result")
    if isinstance(result, dict) and result.get("text") is not None:
        return str(result["text"])

    return ""
