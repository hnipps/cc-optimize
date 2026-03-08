"""Verify evaluator data flow: evaluate_task_run and evaluate_minibatch.

For evaluate_task_run: mock only the RunResult (pointing to a real JSONL tmpfile).
For evaluate_minibatch: feed known TaskResults and verify CompositeScore math.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cc_optimize.adapter.task_runner import RunResult
from cc_optimize.evaluation.evaluator import evaluate_minibatch, evaluate_task_run
from cc_optimize.models.benchmark import (
    BenchmarkTask,
    SignalBaselines,
    SuccessCriterion,
    TaskCategory,
)
from cc_optimize.models.candidate import ConfigCandidate
from cc_optimize.models.result import CompositeScore, TaskResult
from cc_optimize.models.signals import BehavioralSignals


def _baselines() -> SignalBaselines:
    return SignalBaselines(
        efficiency_turn_baseline=10,
        max_looping_severity=0,
        max_tool_error_cascade=2,
        max_repair_frequency=0.15,
    )


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


class TestEvaluateTaskRun:
    def test_plumbs_data_correctly(self, tmp_path):
        """Build known ParsedSession via JSONL, verify TaskResult fields."""
        # Write a JSONL file with known content
        jsonl_path = tmp_path / "trace.jsonl"
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "I'll create the file"},
                {"type": "tool_use", "id": "tc_1", "name": "Write",
                 "input": {"file_path": "/tmp/a.py"}},
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc_1", "content": "ok", "is_error": False},
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Now editing"},
                {"type": "tool_use", "id": "tc_2", "name": "Edit",
                 "input": {"file_path": "/tmp/a.py"}},
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc_2", "content": "ok", "is_error": False},
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "tc_3", "name": "Bash",
                 "input": {"command": "test"}},
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc_3", "content": "error", "is_error": True},
            ]}},
            {"type": "result", "result": "done",
             "usage": {"input_tokens": 1000, "output_tokens": 500}},
        ]
        _write_jsonl(jsonl_path, events)

        # Create a task
        task = BenchmarkTask(
            id="test-task",
            name="Test",
            category=TaskCategory.BUG_FIX,
            prompt="fix",
            repo_path=tmp_path,
            git_ref="HEAD",
            success_criteria=[
                SuccessCriterion(type="file_exists", path="trace.jsonl", description="trace exists"),
            ],
            signal_baselines=_baselines(),
        )

        candidate = ConfigCandidate(id="cand-1", claude_md="test")

        run_result = RunResult(
            session_jsonl_path=jsonl_path,
            worktree_path=tmp_path,
            exit_code=0,
            wall_time_seconds=5.0,
            timed_out=False,
        )

        result = evaluate_task_run(task, candidate, run_result)

        # Verify success
        assert result.success is True  # trace.jsonl exists at tmp_path
        assert result.criteria_results["trace exists"] is True

        # Verify signals are populated
        assert result.signals.turn_count == 3  # 3 assistant blocks
        assert result.signals.tool_error_total_calls == 3
        assert result.signals.tool_error_total_failures == 1  # tc_3 was error

        # Verify tokens from result event
        assert result.tokens_input == 1000
        assert result.tokens_output == 500
        assert result.tokens_total == 1500

        # Verify metrics
        # file_edit_churn: 2 Write/Edit calls, 1 distinct file -> 2.0
        assert result.file_edit_churn == pytest.approx(2.0)
        # tool_error_rate: 1 failure / 3 total = 0.333
        assert result.tool_error_rate == pytest.approx(1.0 / 3.0)


class TestEvaluateMinibatch:
    def _make_task_result(
        self,
        task_id: str,
        criteria: dict[str, bool],
        efficiency: float = 1.0,
        severity: int = 0,
        cascade: int = 0,
        repair_freq: float = 0.0,
        trace_path: Path = Path("/tmp/fake.jsonl"),
    ) -> TaskResult:
        return TaskResult(
            task_id=task_id,
            candidate_id="cand-1",
            success=all(criteria.values()),
            criteria_results=criteria,
            signals=BehavioralSignals(
                efficiency_score=efficiency,
                turn_count=10,
                repetition_count=0,
                repetition_exact_count=0,
                repetition_max_severity=severity,
                tool_error_max_cascade=cascade,
                tool_error_total_failures=0,
                tool_error_total_calls=5,
                repair_frequency=repair_freq,
                repair_count=0,
            ),
            tokens_input=100,
            tokens_output=50,
            tokens_total=150,
            wall_time_seconds=5.0,
            file_edit_churn=1.0,
            tool_error_rate=0.0,
            session_trace_path=trace_path,
        )

    def test_perfect_scores(self):
        """All criteria pass, all signals at baseline -> score near 1.0"""
        baselines = _baselines()
        tasks = [
            BenchmarkTask(
                id=f"t{i}", name=f"Task {i}", category=TaskCategory.BUG_FIX,
                prompt="fix", repo_path=Path("/tmp"), git_ref="HEAD",
                success_criteria=[], signal_baselines=baselines,
            )
            for i in range(2)
        ]
        candidate = ConfigCandidate(id="cand-1", claude_md="test")

        task_results = [
            self._make_task_result("t0", {"c1": True, "c2": True}),
            self._make_task_result("t1", {"c1": True}),
        ]

        result = evaluate_minibatch(tasks, candidate, task_results)

        # Correctness: avg of [1.0, 1.0] = 1.0
        assert result.composite_score.correctness == pytest.approx(1.0)
        # Efficiency: avg of [1.0, 1.0] = 1.0
        assert result.composite_score.efficiency == pytest.approx(1.0)
        # Quality: all at baseline -> 1.0
        assert result.composite_score.conversation_quality == pytest.approx(1.0)
        # Weighted: 0.5*1.0 + 0.25*1.0 + 0.25*1.0 = 1.0
        assert result.composite_score.weighted_scalar() == pytest.approx(1.0)

    def test_mixed_scores(self):
        """Mix of passing and failing criteria."""
        baselines = _baselines()
        tasks = [
            BenchmarkTask(
                id="t0", name="Task 0", category=TaskCategory.BUG_FIX,
                prompt="fix", repo_path=Path("/tmp"), git_ref="HEAD",
                success_criteria=[], signal_baselines=baselines,
            ),
            BenchmarkTask(
                id="t1", name="Task 1", category=TaskCategory.FEATURE,
                prompt="add", repo_path=Path("/tmp"), git_ref="HEAD",
                success_criteria=[], signal_baselines=baselines,
            ),
        ]
        candidate = ConfigCandidate(id="cand-1", claude_md="test")

        task_results = [
            self._make_task_result("t0", {"c1": True, "c2": False}, efficiency=0.5),
            self._make_task_result("t1", {"c1": True, "c2": True, "c3": True}, efficiency=0.8),
        ]

        result = evaluate_minibatch(tasks, candidate, task_results)

        # Correctness: avg of [0.5, 1.0] = 0.75
        assert result.composite_score.correctness == pytest.approx(0.75)
        # Efficiency: avg of [0.5, 0.8] = 0.65
        assert result.composite_score.efficiency == pytest.approx(0.65)

    def test_weighted_scalar_formula(self):
        """Verify weighted_scalar matches spec: (0.5, 0.25, 0.25)."""
        score = CompositeScore(correctness=0.8, efficiency=0.6, conversation_quality=0.4)
        # 0.5*0.8 + 0.25*0.6 + 0.25*0.4 = 0.4 + 0.15 + 0.1 = 0.65
        assert score.weighted_scalar() == pytest.approx(0.65)

    def test_actionable_side_info_nonempty(self):
        """Side info should always be a non-empty string."""
        baselines = _baselines()
        tasks = [
            BenchmarkTask(
                id="t0", name="T", category=TaskCategory.BUG_FIX,
                prompt="fix", repo_path=Path("/tmp"), git_ref="HEAD",
                success_criteria=[], signal_baselines=baselines,
            ),
        ]
        candidate = ConfigCandidate(id="cand-1", claude_md="test")
        task_results = [self._make_task_result("t0", {"c1": True})]

        result = evaluate_minibatch(tasks, candidate, task_results)
        assert isinstance(result.actionable_side_info, str)
        assert len(result.actionable_side_info) > 0

    def test_empty_results(self):
        """Empty task results produce zero scores."""
        candidate = ConfigCandidate(id="cand-1", claude_md="test")
        result = evaluate_minibatch([], candidate, [])
        assert result.composite_score.correctness == 0.0
        assert result.composite_score.efficiency == 0.0
        assert result.composite_score.conversation_quality == 0.0
