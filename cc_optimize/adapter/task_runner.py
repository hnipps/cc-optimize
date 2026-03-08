from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from cc_optimize.adapter.config_applier import apply_config
from cc_optimize.adapter.early_stopper import should_stop
from cc_optimize.config import EarlyStopConfig
from cc_optimize.models.benchmark import BenchmarkTask
from cc_optimize.models.candidate import ConfigCandidate
from cc_optimize.signals.jsonl_parser import parse, ParsedSession

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    session_jsonl_path: Path
    worktree_path: Path
    exit_code: int
    wall_time_seconds: float
    timed_out: bool
    early_stopped: bool = False
    early_stop_reason: str = ""


def _create_worktree(repo_path: Path, worktree_path: Path, git_ref: str) -> bool:
    """Create a git worktree. Returns True on success, False on failure."""
    try:
        subprocess.run(
            ["git", "-C", str(repo_path), "worktree", "add", "--detach", str(worktree_path), git_ref],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to create worktree: %s", e.stderr)
        return False


def _fallback_checkout(repo_path: Path, worktree_path: Path, git_ref: str) -> None:
    """Fallback: clone/copy the repo and checkout the ref."""
    if worktree_path.exists():
        shutil.rmtree(worktree_path)
    worktree_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--shared", str(repo_path), str(worktree_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(worktree_path), "checkout", git_ref],
        check=True,
        capture_output=True,
        text=True,
    )


def run_task(
    task: BenchmarkTask,
    candidate: ConfigCandidate,
    output_dir: Path,
    early_stop_config: EarlyStopConfig | None = None,
) -> RunResult:
    """
    1. Create a git worktree for the task
    2. Apply candidate config to worktree
    3. Run Claude Code CLI with streaming JSON output
    4. Optionally check early stop conditions
    5. Wait for completion or timeout
    6. Return RunResult (do NOT remove worktree)
    """
    timestamp = int(time.time())

    # Create output subdirs
    worktrees_dir = output_dir / "worktrees"
    traces_dir = output_dir / "traces"
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    worktree_path = worktrees_dir / f"{task.id}_{timestamp}"
    trace_path = traces_dir / f"{task.id}_{timestamp}.jsonl"
    stderr_path = traces_dir / f"{task.id}_{timestamp}.stderr"

    # 1. Create worktree (with fallback)
    subprocess.run(
        ["git", "-C", str(task.repo_path), "worktree", "prune"],
        capture_output=True,
        text=True,
    )
    if not _create_worktree(task.repo_path, worktree_path, task.git_ref):
        logger.info("Worktree creation failed, using fallback checkout approach")
        _fallback_checkout(task.repo_path, worktree_path, task.git_ref)

    # 2. Apply candidate config
    apply_config(candidate, worktree_path)

    # 3. Run Claude Code CLI
    cmd = [
        "claude",
        "-p",
        task.prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
    ]

    start_time = time.time()
    timed_out = False
    early_stopped = False
    early_stop_reason = ""
    exit_code = -1

    with open(stderr_path, "w") as stderr_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            cwd=str(worktree_path),
            text=True,
        )

        try:
            line_count = 0
            with open(trace_path, "w") as trace_file:
                assert process.stdout is not None
                for line in process.stdout:
                    trace_file.write(line)
                    trace_file.flush()
                    line_count += 1

                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed > task.timeout_seconds:
                        logger.warning(
                            "Task %s timed out after %.1f seconds", task.id, elapsed
                        )
                        timed_out = True
                        process.kill()
                        break

                    # Check early stop conditions
                    if (
                        early_stop_config is not None
                        and line_count % early_stop_config.check_interval_events == 0
                    ):
                        try:
                            partial_session = parse(trace_path)
                            stop, reason = should_stop(
                                partial_session,
                                task.signal_baselines,
                                early_stop_config,
                            )
                            if stop:
                                logger.info(
                                    "Early stopping task %s: %s", task.id, reason
                                )
                                early_stopped = True
                                early_stop_reason = reason
                                process.terminate()
                                break
                        except Exception:
                            logger.debug(
                                "Early stop check failed (partial parse), continuing"
                            )

            process.wait(timeout=30)
            exit_code = process.returncode

        except subprocess.TimeoutExpired:
            logger.warning("Task %s: process did not exit after terminate, killing", task.id)
            process.kill()
            process.wait()
            exit_code = process.returncode
            timed_out = True

    wall_time = time.time() - start_time

    return RunResult(
        session_jsonl_path=trace_path,
        worktree_path=worktree_path,
        exit_code=exit_code,
        wall_time_seconds=wall_time,
        timed_out=timed_out,
        early_stopped=early_stopped,
        early_stop_reason=early_stop_reason,
    )
