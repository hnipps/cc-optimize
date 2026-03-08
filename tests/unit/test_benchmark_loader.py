from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from cc_optimize.benchmark.loader import load_suite, _load_task
from cc_optimize.benchmark.validator import validate_suite, validate_task, _validate_criterion
from cc_optimize.models.benchmark import (
    BenchmarkSuite,
    BenchmarkTask,
    SignalBaselines,
    SuccessCriterion,
    TaskCategory,
)

CHAINSTATS_SUITE = Path(__file__).parent.parent.parent / "benchmarks" / "chainstats" / "suite.yaml"


class TestLoadSuite:
    def test_load_suite_from_chainstats(self):
        suite = load_suite(CHAINSTATS_SUITE)
        assert suite.name == "chainstats"
        assert len(suite.tasks) == 1

    def test_load_suite_description(self):
        suite = load_suite(CHAINSTATS_SUITE)
        assert "ChainStats" in suite.description

    def test_load_suite_repo_path(self):
        suite = load_suite(CHAINSTATS_SUITE)
        assert suite.repo_path == Path("/home/user/projects/chainstats")


class TestLoadTask:
    def test_load_task_all_fields(self):
        suite = load_suite(CHAINSTATS_SUITE)
        task = suite.tasks[0]

        assert task.id == "go-api-endpoint"
        assert task.name == "Add Go API endpoint for user preferences"
        assert task.category == TaskCategory.FEATURE
        assert "REST endpoint" in task.prompt
        assert task.git_ref == "benchmark/go-api-endpoint-start"
        assert task.timeout_seconds == 600
        assert task.repo_path == Path("/home/user/projects/chainstats")

    def test_load_task_signal_baselines(self):
        suite = load_suite(CHAINSTATS_SUITE)
        task = suite.tasks[0]

        assert task.signal_baselines.efficiency_turn_baseline == 15
        assert task.signal_baselines.max_looping_severity == 0
        assert task.signal_baselines.max_tool_error_cascade == 2
        assert task.signal_baselines.max_repair_frequency == 0.15

    def test_load_task_success_criteria(self):
        suite = load_suite(CHAINSTATS_SUITE)
        task = suite.tasks[0]

        assert len(task.success_criteria) == 5
        assert task.success_criteria[0].type == "command_exit_code"
        assert task.success_criteria[0].command is not None
        assert task.success_criteria[2].type == "file_exists"
        assert task.success_criteria[2].path == "internal/handlers/preferences.go"
        assert task.success_criteria[3].type == "file_contains"
        assert task.success_criteria[3].pattern is not None
        assert task.success_criteria[4].type == "git_diff_excludes"
        assert task.success_criteria[4].paths == ["go.mod", "go.sum"]

    def test_load_task_tags(self):
        suite = load_suite(CHAINSTATS_SUITE)
        task = suite.tasks[0]
        assert set(task.tags) == {"go", "api", "handlers", "feature"}

    def test_invalid_category_raises_error(self, tmp_path):
        task_yaml = tmp_path / "bad_task.yaml"
        task_yaml.write_text(textwrap.dedent("""\
            id: bad-task
            name: "Bad task"
            category: nonexistent_category
            prompt: "Do something"
            git_ref: "main"
            timeout_seconds: 60
            signal_baselines:
              efficiency_turn_baseline: 10
              max_looping_severity: 0
              max_tool_error_cascade: 1
              max_repair_frequency: 0.1
            success_criteria: []
            tags: []
        """))
        with pytest.raises(ValueError):
            _load_task(task_yaml, Path("/tmp/repo"))


class TestValidateSuite:
    def test_valid_suite_no_errors(self):
        suite = load_suite(CHAINSTATS_SUITE)
        errors = validate_suite(suite)
        assert errors == []

    def test_duplicate_task_ids_returns_error(self):
        task = BenchmarkTask(
            id="dupe",
            name="Task",
            category=TaskCategory.FEATURE,
            prompt="Do it",
            repo_path=Path("/tmp"),
            git_ref="main",
            success_criteria=[],
            signal_baselines=SignalBaselines(
                efficiency_turn_baseline=10,
                max_looping_severity=0,
                max_tool_error_cascade=1,
                max_repair_frequency=0.1,
            ),
        )
        suite = BenchmarkSuite(
            name="test",
            repo_path=Path("/tmp"),
            tasks=[task, task],
        )
        errors = validate_suite(suite)
        assert any("Duplicate task ID" in e for e in errors)

    def test_missing_command_for_command_exit_code(self):
        criterion = SuccessCriterion(type="command_exit_code")
        errors = _validate_criterion(criterion)
        assert any("command" in e for e in errors)

    def test_missing_path_for_file_exists(self):
        criterion = SuccessCriterion(type="file_exists")
        errors = _validate_criterion(criterion)
        assert any("path" in e for e in errors)

    def test_missing_paths_for_git_diff_includes(self):
        criterion = SuccessCriterion(type="git_diff_includes")
        errors = _validate_criterion(criterion)
        assert any("paths" in e for e in errors)

    def test_missing_path_for_file_contains(self):
        criterion = SuccessCriterion(type="file_contains")
        errors = _validate_criterion(criterion)
        assert any("path" in e for e in errors)
        assert any("pattern or substring" in e for e in errors)

    def test_file_contains_with_pattern_is_valid(self):
        criterion = SuccessCriterion(type="file_contains", path="foo.py", pattern="bar")
        errors = _validate_criterion(criterion)
        assert errors == []

    def test_file_contains_with_substring_is_valid(self):
        criterion = SuccessCriterion(type="file_contains", path="foo.py", substring="bar")
        errors = _validate_criterion(criterion)
        assert errors == []

    def test_git_diff_excludes_missing_paths(self):
        criterion = SuccessCriterion(type="git_diff_excludes")
        errors = _validate_criterion(criterion)
        assert any("paths" in e for e in errors)

    def test_valid_command_exit_code(self):
        criterion = SuccessCriterion(type="command_exit_code", command="echo hi")
        errors = _validate_criterion(criterion)
        assert errors == []
