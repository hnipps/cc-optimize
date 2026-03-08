from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cc_optimize.adapter.gepa_adapter import ClaudeCodeAdapter
from cc_optimize.config import EarlyStopConfig
from cc_optimize.models.benchmark import (
    BenchmarkSuite,
    BenchmarkTask,
    SignalBaselines,
    SuccessCriterion,
    TaskCategory,
)
from cc_optimize.models.result import TaskResult
from cc_optimize.models.signals import BehavioralSignals


def _make_suite(tmp_path: Path) -> BenchmarkSuite:
    return BenchmarkSuite(
        name="test",
        repo_path=tmp_path,
        tasks=[
            BenchmarkTask(
                id="task-1",
                name="Test task",
                category=TaskCategory.BUG_FIX,
                prompt="Fix the bug",
                repo_path=tmp_path,
                git_ref="main",
                success_criteria=[
                    SuccessCriterion(type="file_exists", path="test.txt", description="file exists")
                ],
                signal_baselines=SignalBaselines(
                    efficiency_turn_baseline=10,
                    max_looping_severity=0,
                    max_tool_error_cascade=2,
                    max_repair_frequency=0.15,
                ),
            )
        ],
    )


def _make_task_result() -> TaskResult:
    return TaskResult(
        task_id="task-1",
        candidate_id="cand-1",
        success=True,
        criteria_results={"file exists": True},
        signals=BehavioralSignals(
            efficiency_score=0.8,
            turn_count=5,
            repetition_count=0,
            repetition_exact_count=0,
            repetition_max_severity=0,
            tool_error_max_cascade=0,
            tool_error_total_failures=0,
            tool_error_total_calls=3,
            repair_frequency=0.0,
            repair_count=0,
        ),
        tokens_input=100,
        tokens_output=50,
        tokens_total=150,
        wall_time_seconds=10.0,
        file_edit_churn=1.0,
        tool_error_rate=0.0,
        session_trace_path=Path("/tmp/trace.jsonl"),
    )


class TestClaudeCodeAdapter:
    @patch("cc_optimize.adapter.gepa_adapter.run_task")
    @patch("cc_optimize.adapter.gepa_adapter.evaluate_task_run")
    def test_evaluate(self, mock_eval, mock_run, tmp_path: Path):
        suite = _make_suite(tmp_path)
        adapter = ClaudeCodeAdapter(
            suite=suite,
            work_dir=tmp_path / "work",
            settings_json={},
        )

        mock_run.return_value = MagicMock(
            worktree_path=tmp_path / "wt",
            session_jsonl_path=tmp_path / "trace.jsonl",
        )
        mock_eval.return_value = _make_task_result()

        candidate = {"claude_md": "Be helpful."}
        batch = [{"task_id": "task-1"}]

        with patch.object(adapter, "_cleanup_worktree"):
            result = adapter.evaluate(batch, candidate)

        assert len(result.scores) == 1
        assert len(result.outputs) == 1
        # Score = 0.5*correctness + 0.25*efficiency + 0.25*quality
        # correctness: 1 criterion passes => 1.0
        # efficiency: signals.efficiency_score = 0.8
        # quality: compute_conversation_quality with efficiency=0.8 =>
        #   0.25*(1.0 + 1.0 + 1.0 + 0.8) = 0.95
        # score = 0.5*1.0 + 0.25*0.8 + 0.25*0.95 = 0.9375
        assert result.scores[0] == pytest.approx(0.9375)
        assert result.objective_scores[0]["correctness"] == pytest.approx(1.0)
        assert result.objective_scores[0]["efficiency"] == pytest.approx(0.8)

    @patch("cc_optimize.adapter.gepa_adapter.run_task")
    @patch("cc_optimize.adapter.gepa_adapter.evaluate_task_run")
    def test_evaluate_unknown_task(self, mock_eval, mock_run, tmp_path: Path):
        suite = _make_suite(tmp_path)
        adapter = ClaudeCodeAdapter(suite=suite, work_dir=tmp_path / "work")
        result = adapter.evaluate([{"task_id": "nonexistent"}], {"claude_md": ""})
        assert len(result.scores) == 0

    def test_make_reflective_dataset(self, tmp_path: Path):
        suite = _make_suite(tmp_path)
        adapter = ClaudeCodeAdapter(suite=suite, work_dir=tmp_path / "work")

        from gepa.core.adapter import EvaluationBatch

        task_result = _make_task_result()
        eval_batch = EvaluationBatch(
            outputs=[task_result],
            scores=[0.8],
            trajectories=[{"task_id": "task-1"}],
            objective_scores=[{"correctness": 0.8}],
        )

        dataset = adapter.make_reflective_dataset(
            candidate={"claude_md": "Be helpful."},
            eval_batch=eval_batch,
            components_to_update=["claude_md"],
        )

        assert "claude_md" in dataset
        assert len(dataset["claude_md"]) == 1
        entry = dataset["claude_md"][0]
        assert "Inputs" in entry
        assert "Generated Outputs" in entry
        assert "Feedback" in entry
        assert "PASS" in entry["Feedback"]

    def test_build_feedback(self, tmp_path: Path):
        suite = _make_suite(tmp_path)
        adapter = ClaudeCodeAdapter(suite=suite, work_dir=tmp_path / "work")
        task_result = _make_task_result()
        feedback = adapter._build_feedback("claude_md", {"claude_md": "test"}, task_result, 0.8)
        assert "Score: 0.80" in feedback
        assert "CLAUDE.md" in feedback
