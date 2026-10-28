"""Base evaluator class and result containers."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..config import EvaluationConfig


@dataclass
class StageResult:
    """Standardized result container for each evaluation stage."""

    stage_name: str
    status: str  # "SUCCESS", "PARTIAL", "FAILED", "SKIPPED", "BLOCKED"
    sample_count: int = 0
    duration_seconds: float = 0.0
    reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    per_sample_results: list[dict[str, Any]] = field(default_factory=list)
    failure_cases: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "sample_count": self.sample_count,
            "duration_seconds": round(self.duration_seconds, 3),
            "reason": self.reason,
            "metrics": self.metrics,
            "failure_case_count": len(self.failure_cases),
            "failure_cases": self.failure_cases[:10],  # preview up to 10 in summary
            "metadata": self.metadata,
        }


class BaseStageEvaluator(ABC):
    """Abstract base class for all pipeline stage evaluators."""

    stage_name: str = "base"

    def execute(self, config: EvaluationConfig) -> StageResult:
        """Run the evaluation with timing, structured logging, and error handling."""
        start_time = time.perf_counter()
        try:
            result = self.run(config)
            result.duration_seconds = time.perf_counter() - start_time
            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            return StageResult(
                stage_name=self.stage_name,
                status="FAILED",
                duration_seconds=duration,
                reason=f"Exception during stage execution: {e}",
                metrics={},
            )

    @abstractmethod
    def run(self, config: EvaluationConfig) -> StageResult:
        """Stage-specific evaluation logic."""
        pass

