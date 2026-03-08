from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, mock_open, patch

import pytest

from cc_optimize.adapter.task_runner import RunResult, run_task
from cc_optimize.config import EarlyStopConfig
from cc_optimize.models.benchmark import BenchmarkTask, SignalBaselines, TaskCategory
from cc_optimize.models.candidate import ConfigCandidate


def _make_task(repo_path: Path, timeout: int = 600) -> BenchmarkTask:
    return BenchmarkTask(
        id="test-task-1",
        name="Test Task",
        category=TaskCategory.BUG_FIX,
        prompt="Fix the bug in main.py",
        repo_path=repo_path,
        git_ref="main",
        success_criteria=[],
        signal_baselines=SignalBaselines(
            efficiency_turn_baseline=10,
            max_looping_severity=3,
            max_tool_error_cascade=5,
            max_repair_frequency=0.5,
        ),
        timeout_seconds=timeout,
    )


def _make_candidate() -> ConfigCandidate:
    return ConfigCandidate(
        id="candidate-1",
        claude_md="# Test config",
    )


class TestRunTaskCreatesPathsCorrectly:
    @patch("cc_optimize.adapter.task_runner.apply_config")
    @patch("cc_optimize.adapter.task_runner.subprocess.Popen")
    @patch("cc_optimize.adapter.task_runner.subprocess.run")
    def test_creates_worktree_and_trace_paths(
        self, mock_run, mock_popen, mock_apply, tmp_path
    ):
        """run_task creates expected worktree and trace paths."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        output_dir = tmp_path / "output"

        # Mock subprocess.run for worktree creation (success)
        mock_run.return_value = MagicMock(returncode=0)

        # Mock Popen for claude CLI
        mock_process = MagicMock()
        mock_process.stdout = iter([
            '{"type": "assistant", "content": [{"type": "text", "text": "hi"}]}\n',
            '{"type": "result", "usage": {"input_tokens": 100, "output_tokens": 50}}\n',
        ])
        mock_process.returncode = 0
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        task = _make_task(repo_path)
        candidate = _make_candidate()

        time_val = [1000000.0]
        def time_side_effect():
            time_val[0] += 0.1
            return time_val[0]

        with patch("cc_optimize.adapter.task_runner.time.time", side_effect=time_side_effect):
            result = run_task(task, candidate, output_dir)

        assert result.worktree_path == output_dir / "worktrees" / "test-task-1_1000000"
        assert result.session_jsonl_path == output_dir / "traces" / "test-task-1_1000000.jsonl"
        assert result.exit_code == 0
        assert result.timed_out is False
        assert result.early_stopped is False

        # Verify worktree creation was attempted
        mock_run.assert_called_once()
        worktree_call = mock_run.call_args
        assert "worktree" in worktree_call[0][0]
        assert "add" in worktree_call[0][0]

        # Verify apply_config was called
        mock_apply.assert_called_once()

        # Verify trace file was written
        assert result.session_jsonl_path.parent.exists()


class TestRunTaskTimeout:
    @patch("cc_optimize.adapter.task_runner.apply_config")
    @patch("cc_optimize.adapter.task_runner.subprocess.Popen")
    @patch("cc_optimize.adapter.task_runner.subprocess.run")
    def test_timeout_handling(self, mock_run, mock_popen, mock_apply, tmp_path):
        """run_task kills the process and sets timed_out=True on timeout."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        output_dir = tmp_path / "output"

        mock_run.return_value = MagicMock(returncode=0)

        # Track time calls: first for timestamp, then start_time, then each
        # elapsed check returns past the timeout
        call_count = 0
        def time_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return 1000000.0
            # After start, each time check is past the 5s timeout
            return 1000010.0

        # Simulate a slow-streaming process that will exceed timeout
        mock_process = MagicMock()
        mock_process.stdout = iter([
            '{"type": "assistant", "content": [{"type": "text", "text": "working..."}]}\n',
            '{"type": "assistant", "content": [{"type": "text", "text": "still working..."}]}\n',
        ])
        mock_process.returncode = -9
        mock_process.wait.return_value = -9
        mock_popen.return_value = mock_process

        task = _make_task(repo_path, timeout=5)
        candidate = _make_candidate()

        with patch("cc_optimize.adapter.task_runner.time.time", side_effect=time_side_effect):
            result = run_task(task, candidate, output_dir)

        assert result.timed_out is True
        mock_process.kill.assert_called()


class TestRunTaskEarlyStop:
    @patch("cc_optimize.adapter.task_runner.apply_config")
    @patch("cc_optimize.adapter.task_runner.subprocess.Popen")
    @patch("cc_optimize.adapter.task_runner.subprocess.run")
    def test_early_stop_integration(
        self, mock_run, mock_popen, mock_apply, tmp_path
    ):
        """run_task terminates process and sets early_stopped when signals exceed thresholds."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        output_dir = tmp_path / "output"

        mock_run.return_value = MagicMock(returncode=0)

        # Generate alternating assistant/user events so the parser creates
        # separate blocks (consecutive assistant events get merged into one).
        # should_stop and parse run on real data here (not mocked).
        repeated_text = "the quick brown fox jumps over the lazy dog again and again"
        assistant_line = f'{{"type": "assistant", "content": [{{"type": "text", "text": "{repeated_text}"}}]}}\n'
        user_line = '{"type": "user", "content": [{"type": "text", "text": "continue"}]}\n'
        lines = []
        for _ in range(20):
            lines.append(assistant_line)
            lines.append(user_line)

        mock_process = MagicMock()
        mock_process.stdout = iter(lines)
        mock_process.returncode = -15
        mock_process.wait.return_value = -15
        mock_popen.return_value = mock_process

        time_val = [1000000.0]
        def time_side_effect():
            time_val[0] += 0.1
            return time_val[0]

        task = _make_task(repo_path, timeout=600)
        candidate = _make_candidate()
        # At check_interval=4, line 4 = 2 blocks (1 pair, severity 1 >= 1)
        early_config = EarlyStopConfig(
            check_interval_events=14,
            max_looping_severity=3,
        )

        with patch("cc_optimize.adapter.task_runner.time.time", side_effect=time_side_effect):
            result = run_task(task, candidate, output_dir, early_stop_config=early_config)

        assert result.early_stopped is True
        assert "Repetition severity" in result.early_stop_reason
        mock_process.terminate.assert_called()


class TestRunTaskWorktreeFallback:
    @patch("cc_optimize.adapter.task_runner.apply_config")
    @patch("cc_optimize.adapter.task_runner.subprocess.Popen")
    @patch("cc_optimize.adapter.task_runner.subprocess.run")
    def test_fallback_on_worktree_failure(self, mock_run, mock_popen, mock_apply, tmp_path):
        """When worktree creation fails, fallback clone/checkout is used."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        output_dir = tmp_path / "output"

        # First call (worktree add) fails, second and third (clone, checkout) succeed
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "git worktree add", stderr="error"),
            MagicMock(returncode=0),  # git clone
            MagicMock(returncode=0),  # git checkout
        ]

        mock_process = MagicMock()
        mock_process.stdout = iter([
            '{"type": "result", "usage": {"input_tokens": 10, "output_tokens": 5}}\n'
        ])
        mock_process.returncode = 0
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        time_val = [1000000.0]
        def time_side_effect():
            time_val[0] += 0.1
            return time_val[0]

        task = _make_task(repo_path)
        candidate = _make_candidate()

        with patch("cc_optimize.adapter.task_runner.time.time", side_effect=time_side_effect):
            result = run_task(task, candidate, output_dir)

        # Should have called worktree add, then clone, then checkout
        assert mock_run.call_count == 3
        assert result.exit_code == 0
