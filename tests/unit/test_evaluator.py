from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from cc_optimize.adapter.task_runner import RunResult
from cc_optimize.evaluation.evaluator import (
    compute_conversation_quality,
    evaluate_minibatch,
    evaluate_task_run,
)
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
        max_looping_severity=2,
        max_tool_error_cascade=2,
        max_repair_frequency=0.3,
    )


def _perfect_signals() -> BehavioralSignals:
    return BehavioralSignals(
        efficiency_score=1.0,
        turn_count=5,
        repetition_count=0,
        repetition_exact_count=0,
        repetition_max_severity=0,
        tool_error_max_cascade=0,
        tool_error_total_failures=0,
        tool_error_total_calls=10,
        repair_frequency=0.0,
        repair_count=0,
    )


def _degraded_signals() -> BehavioralSignals:
    return BehavioralSignals(
        efficiency_score=0.4,
        turn_count=20,
        repetition_count=5,
        repetition_exact_count=3,
        repetition_max_severity=5,  # 3 above baseline of 2
        tool_error_max_cascade=6,  # 4 above baseline of 2
        tool_error_total_failures=8,
        tool_error_total_calls=10,
        repair_frequency=0.8,  # well above baseline of 0.3
        repair_count=4,
    )


class TestComputeConversationQuality:
    def test_perfect_signals(self) -> None:
        quality = compute_conversation_quality(_perfect_signals(), _baselines())
        assert quality == pytest.approx(1.0)

    def test_degraded_signals(self) -> None:
        quality = compute_conversation_quality(_degraded_signals(), _baselines())
        # Repetition: 1.0 - 0.33*3 = 0.01
        # Tool error: 1/(1+0.5*4) = 1/3 = 0.333...
        # Repair: 1 - (0.8-0.3)/(1.0-0.3) = 1 - 0.5/0.7 = ~0.2857
        # Efficiency: 0.4
        # Total: 0.25*(0.01 + 0.333 + 0.2857 + 0.4) = 0.25*1.0287 = ~0.257
        assert quality < 0.5
        assert quality > 0.0

    def test_at_baseline_is_perfect(self) -> None:
        signals = BehavioralSignals(
            efficiency_score=1.0,
            turn_count=10,
            repetition_count=1,
            repetition_exact_count=0,
            repetition_max_severity=2,  # exactly at baseline
            tool_error_max_cascade=2,  # exactly at baseline
            tool_error_total_failures=2,
            tool_error_total_calls=10,
            repair_frequency=0.3,  # exactly at baseline
            repair_count=1,
        )
        quality = compute_conversation_quality(signals, _baselines())
        assert quality == pytest.approx(1.0)


def _make_task(task_id: str = "task-1") -> BenchmarkTask:
    return BenchmarkTask(
        id=task_id,
        name="Test Task",
        category=TaskCategory.BUG_FIX,
        prompt="Fix the bug",
        repo_path=Path("/tmp/fake"),
        git_ref="main",
        success_criteria=[
            SuccessCriterion(type="command_exit_code", command="echo ok", description="echo"),
        ],
        signal_baselines=_baselines(),
    )


def _make_candidate() -> ConfigCandidate:
    return ConfigCandidate(
        id="candidate-1",
        claude_md="Be helpful.",
    )


def _make_task_result(
    task_id: str = "task-1",
    candidate_id: str = "candidate-1",
    success: bool = True,
    signals: BehavioralSignals | None = None,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        candidate_id=candidate_id,
        success=success,
        criteria_results={"criterion_a": success, "criterion_b": True},
        signals=signals or _perfect_signals(),
        tokens_input=1000,
        tokens_output=500,
        tokens_total=1500,
        wall_time_seconds=30.0,
        file_edit_churn=1.5,
        tool_error_rate=0.1,
        session_trace_path=Path("/tmp/trace.jsonl"),
    )


class TestEvaluateMinibatch:
    def test_all_perfect(self) -> None:
        tasks = [_make_task("task-1"), _make_task("task-2")]
        candidate = _make_candidate()
        results = [
            _make_task_result("task-1"),
            _make_task_result("task-2"),
        ]
        mb = evaluate_minibatch(tasks, candidate, results)
        assert mb.candidate_id == "candidate-1"
        # Both tasks fully pass: correctness = 1.0
        assert mb.composite_score.correctness == pytest.approx(1.0)
        assert mb.composite_score.efficiency == pytest.approx(1.0)
        assert mb.composite_score.conversation_quality == pytest.approx(1.0)

    def test_partial_correctness(self) -> None:
        tasks = [_make_task("task-1")]
        candidate = _make_candidate()
        # One criterion passes, one fails: 1/2 = 0.5
        tr = _make_task_result("task-1", success=False)
        tr.criteria_results = {"a": True, "b": False}
        mb = evaluate_minibatch(tasks, candidate, [tr])
        assert mb.composite_score.correctness == pytest.approx(0.5)

    def test_empty_results(self) -> None:
        tasks = []
        candidate = _make_candidate()
        mb = evaluate_minibatch(tasks, candidate, [])
        assert mb.composite_score.correctness == 0.0
        assert mb.composite_score.efficiency == 0.0

    def test_actionable_side_info_contains_scores(self) -> None:
        tasks = [_make_task("task-1")]
        candidate = _make_candidate()
        results = [_make_task_result("task-1")]
        mb = evaluate_minibatch(tasks, candidate, results)
        assert "Weighted score" in mb.actionable_side_info
        assert "Correctness" in mb.actionable_side_info


class TestEvaluateTaskRun:
    def test_with_sample_session(self, tmp_path: Path) -> None:
        # Use the existing sample_session.jsonl fixture
        fixture_path = Path(__file__).parent.parent / "fixtures" / "sample_session.jsonl"

        # Set up a minimal git repo as worktree_path for success checking
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        subprocess.run(["git", "init"], cwd=str(worktree), capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(worktree), capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=str(worktree), capture_output=True, check=True)
        (worktree / "hello.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=str(worktree), capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(worktree), capture_output=True, check=True)

        task = BenchmarkTask(
            id="sample-task",
            name="Sample",
            category=TaskCategory.BUG_FIX,
            prompt="Fix",
            repo_path=Path("/tmp/fake"),
            git_ref="main",
            success_criteria=[
                SuccessCriterion(type="file_exists", path="hello.txt", description="has hello"),
            ],
            signal_baselines=_baselines(),
        )
        candidate = _make_candidate()
        run_result = RunResult(
            session_jsonl_path=fixture_path,
            worktree_path=worktree,
            exit_code=0,
            wall_time_seconds=10.0,
            timed_out=False,
        )

        result = evaluate_task_run(task, candidate, run_result)

        assert result.task_id == "sample-task"
        assert result.candidate_id == "candidate-1"
        assert result.success is True
        assert result.criteria_results["has hello"] is True
        assert result.tokens_input == 1500
        assert result.tokens_output == 350
        assert result.tokens_total == 1850
