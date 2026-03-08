from __future__ import annotations

import logging
import uuid
from pathlib import Path

import gepa
from gepa.core.adapter import EvaluationBatch

from cc_optimize.adapter.config_applier import apply_config
from cc_optimize.adapter.task_runner import RunResult, run_task
from cc_optimize.config import EarlyStopConfig
from cc_optimize.evaluation.evaluator import (
    compute_conversation_quality,
    evaluate_task_run,
)
from cc_optimize.models.benchmark import BenchmarkSuite, BenchmarkTask
from cc_optimize.models.candidate import ConfigCandidate
from cc_optimize.models.result import TaskResult

logger = logging.getLogger(__name__)


class ClaudeCodeAdapter(gepa.GEPAAdapter):
    def __init__(
        self,
        suite: BenchmarkSuite,
        work_dir: Path,
        early_stop_config: EarlyStopConfig | None = None,
        settings_json: dict | None = None,
    ):
        self.suite = suite
        self.work_dir = work_dir
        self.early_stop_config = early_stop_config or EarlyStopConfig()
        self.settings_json = settings_json or {}
        self._task_map: dict[str, BenchmarkTask] = {t.id: t for t in suite.tasks}
        (work_dir / "traces").mkdir(parents=True, exist_ok=True)
        (work_dir / "worktrees").mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        config = ConfigCandidate.from_component_dict(
            candidate, str(uuid.uuid4()), self.settings_json
        )

        outputs: list[TaskResult] = []
        scores: list[float] = []
        trajectories: list[dict] = []
        objective_scores: list[dict[str, float]] = []

        for task_ref in batch:
            task_id = task_ref.get("task_id", "")
            task = self._task_map.get(task_id)
            if task is None:
                logger.warning("Task %s not found in suite, skipping", task_id)
                continue

            run_result = run_task(
                task=task,
                candidate=config,
                output_dir=self.work_dir,
                early_stop_config=self.early_stop_config,
            )

            task_result = evaluate_task_run(task, config, run_result)
            outputs.append(task_result)

            baselines = task.signal_baselines
            criteria_total = len(task_result.criteria_results)
            criteria_passed = sum(task_result.criteria_results.values())
            correctness = criteria_passed / criteria_total if criteria_total > 0 else 0.0
            efficiency = task_result.signals.efficiency_score
            quality = compute_conversation_quality(task_result.signals, baselines)
            score = 0.5 * correctness + 0.25 * efficiency + 0.25 * quality

            scores.append(score)
            objective_scores.append({
                "correctness": correctness,
                "efficiency": efficiency,
                "quality": quality,
            })

            if capture_traces:
                trajectories.append({
                    "task_id": task_id,
                    "session_trace_path": str(task_result.session_trace_path),
                    "success": task_result.success,
                })

            self._cleanup_worktree(run_result.worktree_path, task)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            objective_scores=objective_scores,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        reflective_data: dict[str, list[dict]] = {}
        for component in components_to_update:
            entries = []
            trajectories = eval_batch.trajectories or [{}] * len(eval_batch.outputs)
            for task_result, score, trajectory in zip(
                eval_batch.outputs, eval_batch.scores, trajectories
            ):
                feedback = self._build_feedback(component, candidate, task_result, score)
                entries.append({
                    "Inputs": f"Task: {task_result.task_id}",
                    "Generated Outputs": self._summarize_output(task_result),
                    "Feedback": feedback,
                })
            reflective_data[component] = entries
        return reflective_data

    def _build_feedback(
        self,
        component_name: str,
        candidate: dict[str, str],
        task_result: TaskResult,
        score: float,
    ) -> str:
        signals = task_result.signals
        failed = [
            desc for desc, passed in task_result.criteria_results.items() if not passed
        ]
        failed_str = ", ".join(failed) if failed else "None"

        component_text = candidate.get(component_name, "")
        line_count = len(component_text.splitlines())
        word_count = len(component_text.split())

        if component_name == "claude_md":
            component_context = "This is CLAUDE.md, loaded every session"
        elif component_name.startswith("rule:"):
            first_line = component_text.split("\n", 1)[0] if component_text else ""
            if "PATH_SCOPE:" in first_line:
                scope = first_line.split("PATH_SCOPE:", 1)[1].strip()
                component_context = f"This rule has PATH_SCOPE: {scope}"
            else:
                component_context = "This rule has no PATH_SCOPE annotation"
        elif component_name.startswith("skill:"):
            component_context = "This skill activates on description match"
        else:
            component_context = f"Component type: {component_name}"

        return (
            f"Score: {score:.2f} ({'PASS' if task_result.success else 'FAIL'})\n"
            f"\n"
            f"## Task: {task_result.task_id}\n"
            f"Failed criteria: {failed_str}\n"
            f"\n"
            f"## Signals\n"
            f"- Efficiency: {signals.efficiency_score:.2f} (turns: {signals.turn_count})\n"
            f"- Repetition: severity {signals.repetition_max_severity}\n"
            f"- Tool error cascade: {signals.tool_error_max_cascade}\n"
            f"- Repair frequency: {signals.repair_frequency:.2f}\n"
            f"\n"
            f'## Component "{component_name}" context\n'
            f"- This component is {line_count} lines / {word_count} words\n"
            f"- {component_context}\n"
        )

    def _summarize_output(self, task_result: TaskResult) -> str:
        return (
            f"Task {task_result.task_id}: "
            f"{'PASS' if task_result.success else 'FAIL'}, "
            f"{task_result.signals.turn_count} turns, "
            f"{task_result.tokens_total} tokens"
        )

    def _cleanup_worktree(self, worktree_path: Path, task: BenchmarkTask) -> None:
        import subprocess

        try:
            subprocess.run(
                ["git", "-C", str(task.repo_path), "worktree", "remove", "--force",
                 str(worktree_path)],
                capture_output=True,
                timeout=30,
            )
        except Exception:
            logger.warning("Failed to clean up worktree %s", worktree_path)
