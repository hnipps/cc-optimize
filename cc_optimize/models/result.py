from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cc_optimize.models.signals import BehavioralSignals


@dataclass
class TaskResult:
    task_id: str
    candidate_id: str
    success: bool
    criteria_results: dict[str, bool]
    signals: BehavioralSignals
    tokens_input: int
    tokens_output: int
    tokens_total: int
    wall_time_seconds: float
    file_edit_churn: float
    tool_error_rate: float
    session_trace_path: Path
    llm_judge_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class CompositeScore:
    correctness: float
    efficiency: float
    conversation_quality: float

    def weighted_scalar(
        self, weights: tuple[float, float, float] = (0.5, 0.25, 0.25)
    ) -> float:
        w_c, w_e, w_q = weights
        return w_c * self.correctness + w_e * self.efficiency + w_q * self.conversation_quality


@dataclass
class MinibatchResult:
    candidate_id: str
    task_results: list[TaskResult]
    composite_score: CompositeScore
    actionable_side_info: str
