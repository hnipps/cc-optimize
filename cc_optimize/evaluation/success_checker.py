from __future__ import annotations

import fnmatch
import logging
import re
import subprocess
from pathlib import Path

from cc_optimize.models.benchmark import BenchmarkTask, SuccessCriterion

logger = logging.getLogger(__name__)


def check_success(task: BenchmarkTask, worktree_path: Path) -> dict[str, bool]:
    """
    Run each SuccessCriterion against the post-task worktree state.
    Returns {criterion.description: pass/fail}.

    Types:
    - command_exit_code: run command, check exit code == 0
    - file_exists: check path exists
    - file_contains: check file contains pattern (regex) or substring
    - file_not_contains: check file does NOT contain pattern or substring
    - git_diff_includes: check git diff --name-only includes all criterion.paths (fnmatch)
    - git_diff_excludes: check git diff --name-only has NO matches for criterion.paths

    On any exception: fail that criterion, log error, continue.
    """
    results: dict[str, bool] = {}

    for criterion in task.success_criteria:
        desc = criterion.description or f"{criterion.type}:{criterion.path or criterion.command or ''}"
        try:
            passed = _check_single(criterion, worktree_path)
        except Exception:
            logger.exception("Criterion '%s' raised an exception", desc)
            passed = False
        results[desc] = passed

    return results


def _check_single(criterion: SuccessCriterion, worktree_path: Path) -> bool:
    ctype = criterion.type

    if ctype == "command_exit_code":
        if not criterion.command:
            return False
        result = subprocess.run(
            criterion.command,
            shell=True,
            cwd=str(worktree_path),
            capture_output=True,
        )
        return result.returncode == 0

    elif ctype == "file_exists":
        if not criterion.path:
            return False
        return (worktree_path / criterion.path).exists()

    elif ctype == "file_contains":
        if not criterion.path:
            return False
        file_path = worktree_path / criterion.path
        if not file_path.exists():
            return False
        content = file_path.read_text()
        if criterion.pattern:
            return bool(re.search(criterion.pattern, content, re.DOTALL))
        if criterion.substring:
            return criterion.substring in content
        return False

    elif ctype == "file_not_contains":
        if not criterion.path:
            return False
        file_path = worktree_path / criterion.path
        if not file_path.exists():
            # File doesn't exist, so it can't contain the thing
            return True
        content = file_path.read_text()
        if criterion.pattern:
            return not bool(re.search(criterion.pattern, content, re.DOTALL))
        if criterion.substring:
            return criterion.substring not in content
        return True

    elif ctype == "git_diff_includes":
        changed_files = _git_diff_names(worktree_path)
        if not criterion.paths:
            return False
        for pattern in criterion.paths:
            if not any(fnmatch.fnmatch(f, pattern) for f in changed_files):
                return False
        return True

    elif ctype == "git_diff_excludes":
        changed_files = _git_diff_names(worktree_path)
        if not criterion.paths:
            return True
        for pattern in criterion.paths:
            if any(fnmatch.fnmatch(f, pattern) for f in changed_files):
                return False
        return True

    else:
        logger.warning("Unknown criterion type: %s", ctype)
        return False


def _git_diff_names(worktree_path: Path) -> list[str]:
    """Get list of changed file names from git diff --name-only HEAD."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=str(worktree_path),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning("git diff failed: %s", result.stderr)
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]
