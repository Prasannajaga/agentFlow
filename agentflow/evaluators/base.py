from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentflow.services.run_queries import AgentRunDetail


@dataclass(frozen=True)
class EvaluationResult:
    evaluator_type: str
    passed: bool
    score: float
    summary: str
    expected_json: dict[str, object]
    actual_json: dict[str, object]


class Evaluator(Protocol):
    evaluator_type: str

    def evaluate(self, run: AgentRunDetail, *, expected_text: str) -> EvaluationResult:
        ...
