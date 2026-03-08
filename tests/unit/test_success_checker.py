from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cc_optimize.evaluation.success_checker import check_success
from cc_optimize.models.benchmark import BenchmarkTask, SignalBaselines, SuccessCriterion, TaskCategory


def _make_task(criteria: list[SuccessCriterion]) -> BenchmarkTask:
    return BenchmarkTask(
        id="test-task",
        name="Test Task",
        category=TaskCategory.BUG_FIX,
        prompt="Fix the bug",
        repo_path=Path("/tmp/fake"),
        git_ref="main",
        success_criteria=criteria,
        signal_baselines=SignalBaselines(
            efficiency_turn_baseline=10,
            max_looping_severity=2,
            max_tool_error_cascade=2,
            max_repair_frequency=0.3,
        ),
    )


class TestCommandExitCode:
    def test_passing_command(self, tmp_path: Path) -> None:
        task = _make_task([
            SuccessCriterion(type="command_exit_code", command="echo hello", description="echo"),
        ])
        results = check_success(task, tmp_path)
        assert results["echo"] is True

    def test_failing_command(self, tmp_path: Path) -> None:
        task = _make_task([
            SuccessCriterion(type="command_exit_code", command="false", description="false cmd"),
        ])
        results = check_success(task, tmp_path)
        assert results["false cmd"] is False


class TestFileExists:
    def test_existing_file(self, tmp_path: Path) -> None:
        (tmp_path / "hello.txt").write_text("hi")
        task = _make_task([
            SuccessCriterion(type="file_exists", path="hello.txt", description="file exists"),
        ])
        results = check_success(task, tmp_path)
        assert results["file exists"] is True

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        task = _make_task([
            SuccessCriterion(type="file_exists", path="nope.txt", description="no file"),
        ])
        results = check_success(task, tmp_path)
        assert results["no file"] is False


class TestFileContains:
    def test_substring_present(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("def hello_world():\n    pass\n")
        task = _make_task([
            SuccessCriterion(
                type="file_contains",
                path="code.py",
                substring="hello_world",
                description="has hello",
            ),
        ])
        results = check_success(task, tmp_path)
        assert results["has hello"] is True

    def test_substring_missing(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("def foo():\n    pass\n")
        task = _make_task([
            SuccessCriterion(
                type="file_contains",
                path="code.py",
                substring="hello_world",
                description="missing hello",
            ),
        ])
        results = check_success(task, tmp_path)
        assert results["missing hello"] is False

    def test_pattern_matching(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("version = '1.2.3'\n")
        task = _make_task([
            SuccessCriterion(
                type="file_contains",
                path="code.py",
                pattern=r"version\s*=\s*'[\d.]+'",
                description="version pattern",
            ),
        ])
        results = check_success(task, tmp_path)
        assert results["version pattern"] is True

    def test_pattern_not_matching(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("name = 'foo'\n")
        task = _make_task([
            SuccessCriterion(
                type="file_contains",
                path="code.py",
                pattern=r"version\s*=\s*'[\d.]+'",
                description="no version",
            ),
        ])
        results = check_success(task, tmp_path)
        assert results["no version"] is False


class TestFileNotContains:
    def test_absent_passes(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("def foo():\n    pass\n")
        task = _make_task([
            SuccessCriterion(
                type="file_not_contains",
                path="code.py",
                substring="debugger",
                description="no debugger",
            ),
        ])
        results = check_success(task, tmp_path)
        assert results["no debugger"] is True

    def test_present_fails(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("import pdb; pdb.set_trace()\n")
        task = _make_task([
            SuccessCriterion(
                type="file_not_contains",
                path="code.py",
                substring="pdb",
                description="no pdb",
            ),
        ])
        results = check_success(task, tmp_path)
        assert results["no pdb"] is False


def _init_git_repo(path: Path) -> None:
    """Initialize a git repo, create a file, and commit."""
    subprocess.run(["git", "init"], cwd=str(path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(path), capture_output=True, check=True)
    (path / "src").mkdir()
    (path / "src" / "main.py").write_text("print('hello')\n")
    (path / "README.md").write_text("# Project\n")
    subprocess.run(["git", "add", "."], cwd=str(path), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(path), capture_output=True, check=True)


class TestGitDiffIncludes:
    def test_includes_matching(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        # Modify the file
        (tmp_path / "src" / "main.py").write_text("print('updated')\n")
        task = _make_task([
            SuccessCriterion(
                type="git_diff_includes",
                paths=["src/*.py"],
                description="diff includes src py",
            ),
        ])
        results = check_success(task, tmp_path)
        assert results["diff includes src py"] is True

    def test_includes_not_matching(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        # Modify the file
        (tmp_path / "src" / "main.py").write_text("print('updated')\n")
        task = _make_task([
            SuccessCriterion(
                type="git_diff_includes",
                paths=["tests/*.py"],
                description="diff includes tests",
            ),
        ])
        results = check_success(task, tmp_path)
        assert results["diff includes tests"] is False


class TestGitDiffExcludes:
    def test_excludes_no_match(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        (tmp_path / "src" / "main.py").write_text("print('updated')\n")
        task = _make_task([
            SuccessCriterion(
                type="git_diff_excludes",
                paths=["*.md"],
                description="no md changes",
            ),
        ])
        results = check_success(task, tmp_path)
        assert results["no md changes"] is True

    def test_excludes_has_match(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        (tmp_path / "README.md").write_text("# Updated\n")
        task = _make_task([
            SuccessCriterion(
                type="git_diff_excludes",
                paths=["*.md"],
                description="no md changes",
            ),
        ])
        results = check_success(task, tmp_path)
        assert results["no md changes"] is False
