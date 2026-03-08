from __future__ import annotations

from cc_optimize.adapter.task_runner import RunResult
from cc_optimize.evaluation.metrics import compute_coding_metrics
from cc_optimize.evaluation.success_checker import check_success
from cc_optimize.models.benchmark import BenchmarkTask, SignalBaselines
from cc_optimize.models.candidate import ConfigCandidate
from cc_optimize.models.result import CompositeScore, MinibatchResult, TaskResult
from cc_optimize.models.signals import BehavioralSignals
from cc_optimize.signals.compute import compute_all_signals
from cc_optimize.signals.jsonl_parser import parse


def compute_conversation_quality(signals: BehavioralSignals, baselines: SignalBaselines) -> float:
    """
    0.0-1.0 composite of signal health. Each signal contributes equally (0.25 weight).

    - Repetition: 1.0 if severity <= baseline max, decreasing by 0.33 per level above
    - Tool error: 1.0 if max_cascade <= baseline, then 1 / (1 + 0.5 * excess)
    - Repair: 1.0 if frequency <= baseline max, then linear decrease to 0 at frequency=1.0
    - Efficiency: the efficiency_score itself
    """
    # Repetition component
    severity_excess = signals.repetition_max_severity - baselines.max_looping_severity
    if severity_excess <= 0:
        repetition_score = 1.0
    else:
        repetition_score = max(0.0, 1.0 - 0.33 * severity_excess)

    # Tool error component
    cascade_excess = signals.tool_error_max_cascade - baselines.max_tool_error_cascade
    if cascade_excess <= 0:
        tool_error_score = 1.0
    else:
        tool_error_score = 1.0 / (1.0 + 0.5 * cascade_excess)

    # Repair component
    if signals.repair_frequency <= baselines.max_repair_frequency:
        repair_score = 1.0
    else:
        # Linear decrease from baseline to 1.0
        remaining_range = 1.0 - baselines.max_repair_frequency
        if remaining_range <= 0:
            repair_score = 0.0
        else:
            overshoot = signals.repair_frequency - baselines.max_repair_frequency
            repair_score = max(0.0, 1.0 - overshoot / remaining_range)

    # Efficiency component
    efficiency_score = signals.efficiency_score

    return 0.25 * repetition_score + 0.25 * tool_error_score + 0.25 * repair_score + 0.25 * efficiency_score


def evaluate_task_run(
    task: BenchmarkTask,
    candidate: ConfigCandidate,
    run_result: RunResult,
) -> TaskResult:
    """
    1. Parse session from run_result.session_jsonl_path
    2. Compute signals
    3. Check success
    4. Compute metrics
    5. Assemble TaskResult
    """
    session = parse(run_result.session_jsonl_path)
    signals = compute_all_signals(session, task.signal_baselines)

    if run_result.early_stopped:
        signals.early_stopped = True
        signals.early_stop_reason = run_result.early_stop_reason

    criteria_results = check_success(task, run_result.worktree_path)
    metrics = compute_coding_metrics(session)

    success = all(criteria_results.values()) if criteria_results else False

    return TaskResult(
        task_id=task.id,
        candidate_id=candidate.id,
        success=success,
        criteria_results=criteria_results,
        signals=signals,
        tokens_input=metrics["tokens_input"],
        tokens_output=metrics["tokens_output"],
        tokens_total=metrics["tokens_total"],
        wall_time_seconds=run_result.wall_time_seconds,
        file_edit_churn=metrics["file_edit_churn"],
        tool_error_rate=metrics["tool_error_rate"],
        session_trace_path=run_result.session_jsonl_path,
    )


def evaluate_minibatch(
    tasks: list[BenchmarkTask],
    candidate: ConfigCandidate,
    task_results: list[TaskResult],
    score_weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> MinibatchResult:
    """
    1. correctness = average of (criteria_passed / criteria_total) across tasks
    2. efficiency = average of signals.efficiency_score
    3. conversation_quality = average of compute_conversation_quality()
    4. Build CompositeScore
    5. Generate actionable_side_info
    6. Return MinibatchResult
    """
    if not task_results:
        return MinibatchResult(
            candidate_id=candidate.id,
            task_results=[],
            composite_score=CompositeScore(correctness=0.0, efficiency=0.0, conversation_quality=0.0),
            actionable_side_info="No task results to evaluate.",
        )

    # Build task lookup for baselines
    task_map = {t.id: t for t in tasks}

    # 1. Correctness
    correctness_scores: list[float] = []
    for tr in task_results:
        total = len(tr.criteria_results)
        if total > 0:
            passed = sum(1 for v in tr.criteria_results.values() if v)
            correctness_scores.append(passed / total)
        else:
            correctness_scores.append(0.0)
    correctness = sum(correctness_scores) / len(correctness_scores)

    # 2. Efficiency
    efficiency_scores = [tr.signals.efficiency_score for tr in task_results]
    efficiency = sum(efficiency_scores) / len(efficiency_scores)

    # 3. Conversation quality
    quality_scores: list[float] = []
    for tr in task_results:
        task = task_map.get(tr.task_id)
        if task:
            quality_scores.append(compute_conversation_quality(tr.signals, task.signal_baselines))
        else:
            quality_scores.append(0.0)
    conversation_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    composite = CompositeScore(
        correctness=correctness,
        efficiency=efficiency,
        conversation_quality=conversation_quality,
    )

    # 5. Actionable side info
    side_info_parts: list[str] = []
    weighted = composite.weighted_scalar(score_weights)
    side_info_parts.append(f"Weighted score: {weighted:.3f}")
    side_info_parts.append(f"Correctness: {correctness:.3f}, Efficiency: {efficiency:.3f}, Quality: {conversation_quality:.3f}")

    # Flag notable issues
    for tr in task_results:
        if tr.signals.repetition_max_severity > 2:
            side_info_parts.append(f"Task {tr.task_id}: high repetition severity ({tr.signals.repetition_max_severity})")
        if tr.signals.tool_error_max_cascade > 3:
            side_info_parts.append(f"Task {tr.task_id}: tool error cascade ({tr.signals.tool_error_max_cascade})")
        if tr.signals.early_stopped:
            side_info_parts.append(f"Task {tr.task_id}: early stopped ({tr.signals.early_stop_reason})")
        if not tr.success:
            failed = [k for k, v in tr.criteria_results.items() if not v]
            side_info_parts.append(f"Task {tr.task_id}: failed criteria: {failed}")

    actionable_side_info = "\n".join(side_info_parts)

    return MinibatchResult(
        candidate_id=candidate.id,
        task_results=task_results,
        composite_score=composite,
        actionable_side_info=actionable_side_info,
    )
