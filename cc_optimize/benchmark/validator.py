from __future__ import annotations

from cc_optimize.models.benchmark import BenchmarkSuite, BenchmarkTask, SuccessCriterion


def validate_suite(suite: BenchmarkSuite) -> list[str]:
    """
    Validate a benchmark suite. Returns list of error messages (empty = valid).

    Checks:
    - All task IDs are unique
    - All success criteria have required fields for their type
    - signal_baselines has all required fields (enforced by dataclass)
    """
    errors: list[str] = []

    # Check for duplicate task IDs
    seen_ids: set[str] = set()
    for task in suite.tasks:
        if task.id in seen_ids:
            errors.append(f"Duplicate task ID: {task.id}")
        seen_ids.add(task.id)

    # Validate each task
    for task in suite.tasks:
        errors.extend(validate_task(task))

    return errors


def validate_task(task: BenchmarkTask) -> list[str]:
    """Validate a single task."""
    errors: list[str] = []
    for criterion in task.success_criteria:
        errors.extend(_validate_criterion(criterion))
    return errors


def _validate_criterion(criterion: SuccessCriterion) -> list[str]:
    """Validate a single success criterion has required fields for its type.

    Required fields by type:
    - command_exit_code: command
    - file_exists: path
    - file_contains: path + (pattern or substring)
    - file_not_contains: path + (pattern or substring)
    - git_diff_includes: paths
    - git_diff_excludes: paths
    """
    errors: list[str] = []
    ctype = criterion.type

    if ctype == "command_exit_code":
        if not criterion.command:
            errors.append(
                f"Criterion '{ctype}' missing required field: command"
            )
    elif ctype == "file_exists":
        if not criterion.path:
            errors.append(
                f"Criterion '{ctype}' missing required field: path"
            )
    elif ctype in ("file_contains", "file_not_contains"):
        if not criterion.path:
            errors.append(
                f"Criterion '{ctype}' missing required field: path"
            )
        if not criterion.pattern and not criterion.substring:
            errors.append(
                f"Criterion '{ctype}' missing required field: pattern or substring"
            )
    elif ctype in ("git_diff_includes", "git_diff_excludes"):
        if not criterion.paths:
            errors.append(
                f"Criterion '{ctype}' missing required field: paths"
            )

    return errors
