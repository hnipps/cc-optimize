from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TaskCategory(Enum):
    BUG_FIX = "bug_fix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    DATABASE = "database"
    CROSS_STACK = "cross_stack"
    TEST_WRITING = "test_writing"
    DOCUMENTATION = "documentation"


@dataclass
class SignalBaselines:
    efficiency_turn_baseline: int
    max_looping_severity: int
    max_tool_error_cascade: int
    max_repair_frequency: float


@dataclass
class SuccessCriterion:
    type: str
    command: str | None = None
    path: str | None = None
    pattern: str | None = None
    substring: str | None = None
    paths: list[str] | None = None
    description: str = ""


@dataclass
class BenchmarkTask:
    id: str
    name: str
    category: TaskCategory
    prompt: str
    repo_path: Path
    git_ref: str
    success_criteria: list[SuccessCriterion]
    signal_baselines: SignalBaselines
    timeout_seconds: int = 600
    tags: list[str] = field(default_factory=list)


@dataclass
class BenchmarkSuite:
    name: str
    repo_path: Path
    tasks: list[BenchmarkTask]
    description: str = ""
