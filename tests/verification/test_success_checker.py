"""Verify all 6 success criterion types from spec Section 4.7.

Uses real tmp git repos for git_diff tests and real filesystem for file tests.
Only mocking is avoided — we use actual subprocess and filesystem operations.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cc_optimize.evaluation.success_checker import check_success
from cc_optimize.models.benchmark import (
    BenchmarkTask,
    SignalBaselines,
    SuccessCriterion,
    TaskCategory,
)


def _baselines() -> SignalBaselines:
    return SignalBaselines(
        efficiency_turn_baseline=10,
        max_looping_severity=0,
        max_tool_error_cascade=2,
        max_repair_frequency=0.15,
    )


def _make_task(criteria: list[SuccessCriterion], repo_path: Path) -> BenchmarkTask:
    return BenchmarkTask(
        id="test-task",
        name="Test Task",
        category=TaskCategory.BUG_FIX,
        prompt="fix the bug",
        repo_path=repo_path,
        git_ref="HEAD",
        success_criteria=criteria,
        signal_baselines=_baselines(),
    )


def _init_git_repo(path: Path) -> None:
    """Initialize a git repo with an initial commit."""
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        check=True, capture_output=True,
    )
    # Create an initial file and commit
    (path / "initial.txt").write_text("initial content\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        check=True, capture_output=True,
    )


class TestCommandExitCode:
    def test_passing_command(self, tmp_path):
        _init_git_repo(tmp_path)
        criteria = [SuccessCriterion(type="command_exit_code", command="echo hello", description="echo works")]
        task = _make_task(criteria, tmp_path)
        results = check_success(task, tmp_path)
        assert results["echo works"] is True

    def test_failing_command(self, tmp_path):
        _init_git_repo(tmp_path)
        criteria = [SuccessCriterion(type="command_exit_code", command="false", description="false fails")]
        task = _make_task(criteria, tmp_path)
        results = check_success(task, tmp_path)
        assert results["false fails"] is False


class TestFileExists:
    def test_existing_file(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "present.txt").write_text("here")
        criteria = [SuccessCriterion(type="file_exists", path="present.txt", description="file present")]
        task = _make_task(criteria, tmp_path)
        results = check_success(task, tmp_path)
        assert results["file present"] is True

    def test_missing_file(self, tmp_path):
        _init_git_repo(tmp_path)
        criteria = [SuccessCriterion(type="file_exists", path="absent.txt", description="file missing")]
        task = _make_task(criteria, tmp_path)
        results = check_success(task, tmp_path)
        assert results["file missing"] is False


class TestFileContains:
    def test_pattern_match(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "code.go").write_text("func GetPreferences(w http.ResponseWriter) {}")
        criteria = [
            SuccessCriterion(
                type="file_contains",
                path="code.go",
                pattern=r"func.*GetPreferences.*http\.ResponseWriter",
                description="handler signature",
            )
        ]
        task = _make_task(criteria, tmp_path)
        results = check_success(task, tmp_path)
        assert results["handler signature"] is True

    def test_pattern_no_match(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "code.go").write_text("func main() {}")
        criteria = [
            SuccessCriterion(
                type="file_contains",
                path="code.go",
                pattern=r"func.*GetPreferences",
                description="handler missing",
            )
        ]
        task = _make_task(criteria, tmp_path)
        results = check_success(task, tmp_path)
        assert results["handler missing"] is False

    def test_substring_match(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "readme.md").write_text("# Hello World\nThis is a test.")
        criteria = [
            SuccessCriterion(
                type="file_contains",
                path="readme.md",
                substring="Hello World",
                description="has title",
            )
        ]
        task = _make_task(criteria, tmp_path)
        results = check_success(task, tmp_path)
        assert results["has title"] is True


class TestFileNotContains:
    def test_pattern_absent(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "code.go").write_text("func main() {}")
        criteria = [
            SuccessCriterion(
                type="file_not_contains",
                path="code.go",
                pattern=r"panic\(",
                description="no panics",
            )
        ]
        task = _make_task(criteria, tmp_path)
        results = check_success(task, tmp_path)
        assert results["no panics"] is True

    def test_pattern_present(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "code.go").write_text('func bad() { panic("oops") }')
        criteria = [
            SuccessCriterion(
                type="file_not_contains",
                path="code.go",
                pattern=r"panic\(",
                description="has panics",
            )
        ]
        task = _make_task(criteria, tmp_path)
        results = check_success(task, tmp_path)
        assert results["has panics"] is False

    def test_substring_absent(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "code.py").write_text("print('hello')")
        criteria = [
            SuccessCriterion(
                type="file_not_contains",
                path="code.py",
                substring="import os",
                description="no os import",
            )
        ]
        task = _make_task(criteria, tmp_path)
        results = check_success(task, tmp_path)
        assert results["no os import"] is True


class TestGitDiffIncludes:
    def test_changed_file_detected(self, tmp_path):
        _init_git_repo(tmp_path)
        # Modify a tracked file
        (tmp_path / "initial.txt").write_text("modified content\n")
        criteria = [
            SuccessCriterion(
                type="git_diff_includes",
                paths=["initial.txt"],
                description="initial changed",
            )
        ]
        task = _make_task(criteria, tmp_path)
        results = check_success(task, tmp_path)
        assert results["initial changed"] is True

    def test_unchanged_file_not_detected(self, tmp_path):
        _init_git_repo(tmp_path)
        # Don't modify anything
        criteria = [
            SuccessCriterion(
                type="git_diff_includes",
                paths=["initial.txt"],
                description="initial not changed",
            )
        ]
        task = _make_task(criteria, tmp_path)
        results = check_success(task, tmp_path)
        assert results["initial not changed"] is False


class TestGitDiffExcludes:
    def test_excluded_path_not_changed(self, tmp_path):
        _init_git_repo(tmp_path)
        # Create and modify a different file
        (tmp_path / "other.txt").write_text("new")
        subprocess.run(["git", "-C", str(tmp_path), "add", "other.txt"], check=True, capture_output=True)
        (tmp_path / "other.txt").write_text("changed")
        criteria = [
            SuccessCriterion(
                type="git_diff_excludes",
                paths=["initial.txt"],
                description="initial untouched",
            )
        ]
        task = _make_task(criteria, tmp_path)
        results = check_success(task, tmp_path)
        assert results["initial untouched"] is True

    def test_excluded_path_changed(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "initial.txt").write_text("changed!\n")
        criteria = [
            SuccessCriterion(
                type="git_diff_excludes",
                paths=["initial.txt"],
                description="initial should be clean",
            )
        ]
        task = _make_task(criteria, tmp_path)
        results = check_success(task, tmp_path)
        assert results["initial should be clean"] is False


class TestExceptionHandling:
    def test_exception_fails_criterion_but_continues(self, tmp_path):
        """An exception in one criterion should not prevent checking others."""
        _init_git_repo(tmp_path)
        (tmp_path / "exists.txt").write_text("here")
        criteria = [
            # This will raise due to timeout (command sleeps for 200s but subprocess has no timeout set;
            # however, accessing a nonexistent file for file_contains will fail gracefully)
            SuccessCriterion(
                type="file_contains",
                path="nonexistent_dir/deep/file.txt",
                substring="something",
                description="bad criterion",
            ),
            SuccessCriterion(
                type="file_exists",
                path="exists.txt",
                description="good criterion",
            ),
        ]
        task = _make_task(criteria, tmp_path)
        results = check_success(task, tmp_path)
        # Bad criterion fails (file doesn't exist)
        assert results["bad criterion"] is False
        # Good criterion still runs and passes
        assert results["good criterion"] is True

    def test_multiple_criteria_all_evaluated(self, tmp_path):
        """All criteria are evaluated even when some fail."""
        _init_git_repo(tmp_path)
        criteria = [
            SuccessCriterion(type="file_exists", path="nope1.txt", description="c1"),
            SuccessCriterion(type="file_exists", path="nope2.txt", description="c2"),
            SuccessCriterion(type="command_exit_code", command="true", description="c3"),
        ]
        task = _make_task(criteria, tmp_path)
        results = check_success(task, tmp_path)
        assert len(results) == 3
        assert results["c1"] is False
        assert results["c2"] is False
        assert results["c3"] is True
