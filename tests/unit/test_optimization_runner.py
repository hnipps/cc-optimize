from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from cc_optimize.config import OptimizationConfig
from cc_optimize.models.benchmark import (
    BenchmarkSuite,
    BenchmarkTask,
    SignalBaselines,
    SuccessCriterion,
    TaskCategory,
)
from cc_optimize.models.candidate import ConfigCandidate


def _make_suite(tmp_path: Path) -> BenchmarkSuite:
    return BenchmarkSuite(
        name="test",
        repo_path=tmp_path,
        tasks=[
            BenchmarkTask(
                id="t1",
                name="Task 1",
                category=TaskCategory.BUG_FIX,
                prompt="Fix it",
                repo_path=tmp_path,
                git_ref="main",
                success_criteria=[
                    SuccessCriterion(type="file_exists", path="a.txt", description="exists")
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


class TestOptimizationRunner:
    @patch("cc_optimize.optimization.runner.gepa")
    def test_passes_correct_args_to_gepa(self, mock_gepa, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.best_candidate = {"claude_md": "Optimized config."}
        mock_result.candidates = [{"claude_md": "Optimized config."}]
        mock_result.val_aggregate_scores = [0.5, 0.7, 0.8]
        mock_result.total_metric_calls = 10
        mock_gepa.optimize.return_value = mock_result

        from cc_optimize.optimization.runner import run_optimization

        suite = _make_suite(tmp_path)
        seed = ConfigCandidate(id="seed", claude_md="Be helpful.")
        config = OptimizationConfig(max_metric_calls=10)

        run_optimization(suite, seed, config, work_dir=tmp_path / "work")

        mock_gepa.optimize.assert_called_once()
        call_kwargs = mock_gepa.optimize.call_args
        assert call_kwargs.kwargs["seed_candidate"] == {"claude_md": "Be helpful."}
        assert call_kwargs.kwargs["max_metric_calls"] == 10
        assert call_kwargs.kwargs["trainset"] == [{"task_id": "t1"}]
        assert call_kwargs.kwargs["valset"] == [{"task_id": "t1"}]
        assert call_kwargs.kwargs["reflection_lm"] == config.reflection_model

    @patch("cc_optimize.optimization.runner.ClaudeCodeAdapter")
    @patch("cc_optimize.optimization.runner.gepa")
    def test_report_token_fields_from_adapter(self, mock_gepa, mock_adapter_cls, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.best_candidate = {"claude_md": "Optimized."}
        mock_result.candidates = []
        mock_result.val_aggregate_scores = [0.5]
        mock_result.total_metric_calls = 5
        mock_gepa.optimize.return_value = mock_result

        mock_adapter = MagicMock()
        mock_adapter.total_tokens_consumed = 42000
        mock_adapter.total_early_stop_tokens_saved = 8000
        mock_adapter_cls.return_value = mock_adapter

        from cc_optimize.optimization.runner import run_optimization

        suite = _make_suite(tmp_path)
        seed = ConfigCandidate(id="seed", claude_md="Be helpful.")
        config = OptimizationConfig(max_metric_calls=5)

        report = run_optimization(suite, seed, config, work_dir=tmp_path / "work")

        assert report.total_tokens_consumed == 42000
        assert report.tokens_saved_by_early_stopping == 8000

    @patch("cc_optimize.optimization.runner.gepa")
    def test_converts_result_to_report(self, mock_gepa, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.best_candidate = {"claude_md": "Optimized."}
        mock_result.candidates = [{"claude_md": "Optimized."}]
        mock_result.val_aggregate_scores = [0.5, 0.7, 0.8]
        mock_result.total_metric_calls = 10
        mock_gepa.optimize.return_value = mock_result

        from cc_optimize.optimization.runner import run_optimization

        suite = _make_suite(tmp_path)
        seed = ConfigCandidate(id="seed", claude_md="Be helpful.")
        config = OptimizationConfig(max_metric_calls=10)

        report = run_optimization(suite, seed, config, work_dir=tmp_path / "work")

        assert report.seed_candidate is seed
        assert report.best_candidate.claude_md == "Optimized."
        assert report.optimization_trace == [
            {"iteration": 0, "score": 0.5},
            {"iteration": 1, "score": 0.7},
            {"iteration": 2, "score": 0.8},
        ]
        assert len(report.pareto_front) == 1
        assert report.wall_time_seconds > 0
