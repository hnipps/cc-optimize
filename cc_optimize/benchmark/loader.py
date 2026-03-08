from __future__ import annotations

from pathlib import Path

import yaml

from cc_optimize.models.benchmark import (
    BenchmarkSuite,
    BenchmarkTask,
    SignalBaselines,
    SuccessCriterion,
    TaskCategory,
)


def load_suite(suite_path: Path) -> BenchmarkSuite:
    """
    1. Parse suite.yaml
    2. Resolve task file paths relative to suite.yaml's directory
    3. Load and validate each task YAML
    4. Return BenchmarkSuite
    """
    suite_path = Path(suite_path)
    with open(suite_path) as f:
        data = yaml.safe_load(f)

    suite_dir = suite_path.parent
    repo_path = Path(data["repo_path"])

    tasks: list[BenchmarkTask] = []
    for task_rel in data.get("tasks", []):
        task_path = suite_dir / task_rel
        tasks.append(_load_task(task_path, repo_path))

    return BenchmarkSuite(
        name=data["name"],
        repo_path=repo_path,
        tasks=tasks,
        description=data.get("description", ""),
    )


def _load_task(task_path: Path, repo_path: Path) -> BenchmarkTask:
    """Load a single task YAML file."""
    with open(task_path) as f:
        data = yaml.safe_load(f)

    category = TaskCategory(data["category"])

    baselines_data = data["signal_baselines"]
    signal_baselines = SignalBaselines(
        efficiency_turn_baseline=baselines_data["efficiency_turn_baseline"],
        max_looping_severity=baselines_data["max_looping_severity"],
        max_tool_error_cascade=baselines_data["max_tool_error_cascade"],
        max_repair_frequency=baselines_data["max_repair_frequency"],
    )

    criteria: list[SuccessCriterion] = []
    for sc in data.get("success_criteria", []):
        criteria.append(
            SuccessCriterion(
                type=sc["type"],
                command=sc.get("command"),
                path=sc.get("path"),
                pattern=sc.get("pattern"),
                substring=sc.get("substring"),
                paths=sc.get("paths"),
                description=sc.get("description", ""),
            )
        )

    return BenchmarkTask(
        id=data["id"],
        name=data["name"],
        category=category,
        prompt=data["prompt"],
        repo_path=repo_path,
        git_ref=data["git_ref"],
        success_criteria=criteria,
        signal_baselines=signal_baselines,
        timeout_seconds=data.get("timeout_seconds", 600),
        tags=data.get("tags", []),
    )
