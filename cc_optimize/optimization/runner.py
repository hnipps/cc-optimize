from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import gepa

from cc_optimize.adapter.gepa_adapter import ClaudeCodeAdapter
from cc_optimize.config import EarlyStopConfig, OptimizationConfig
from cc_optimize.models.benchmark import BenchmarkSuite
from cc_optimize.models.candidate import ConfigCandidate
from cc_optimize.optimization.report import OptimizationReport

logger = logging.getLogger(__name__)


def run_optimization(
    suite: BenchmarkSuite,
    seed_config: ConfigCandidate,
    config: OptimizationConfig,
    work_dir: Path | None = None,
    early_stop_config: EarlyStopConfig | None = None,
) -> OptimizationReport:
    work_dir = work_dir or Path("/tmp/cc-optimize")
    work_dir.mkdir(parents=True, exist_ok=True)

    seed_dict = seed_config.to_component_dict()

    trainset = [{"task_id": t.id} for t in suite.tasks]
    valset = trainset

    adapter = ClaudeCodeAdapter(
        suite=suite,
        work_dir=work_dir,
        early_stop_config=early_stop_config or EarlyStopConfig(),
        settings_json=seed_config.settings_json,
        num_trials=config.num_trials,
    )

    start_time = time.time()

    result = gepa.optimize(
        seed_candidate=seed_dict,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=config.reflection_model,
        max_metric_calls=config.max_metric_calls,
    )

    wall_time = time.time() - start_time

    best_candidate = ConfigCandidate.from_component_dict(
        result.best_candidate,
        str(uuid.uuid4()),
        seed_config.settings_json,
    )

    pareto_front = []
    for i, cand_dict in enumerate(getattr(result, "candidates", [])):
        if isinstance(cand_dict, dict):
            pareto_front.append(
                ConfigCandidate.from_component_dict(
                    cand_dict, str(uuid.uuid4()), seed_config.settings_json
                )
            )

    val_scores = getattr(result, "val_aggregate_scores", [])
    optimization_trace = []
    for i, score in enumerate(val_scores):
        optimization_trace.append({
            "iteration": i,
            "score": score,
        })

    # Build per-task comparison: seed scores vs best scores
    per_task_comparison: dict[str, dict] = {}
    val_subscores = getattr(result, "val_subscores", [])
    raw_best_idx = getattr(result, "best_idx", None)
    best_idx = raw_best_idx if isinstance(raw_best_idx, int) else (len(val_scores) - 1 if val_scores else 0)
    if val_subscores:
        seed_subscores = val_subscores[0] if len(val_subscores) > 0 else {}
        best_subscores = val_subscores[best_idx] if best_idx < len(val_subscores) else {}
        all_task_ids = set(seed_subscores) | set(best_subscores)
        for task_id in all_task_ids:
            seed_score = seed_subscores.get(task_id, 0.0)
            best_score = best_subscores.get(task_id, 0.0)
            per_task_comparison[str(task_id)] = {
                "seed_score": seed_score,
                "best_score": best_score,
            }

    # Build signal improvements from per-objective aggregate scores
    signal_improvements: dict[str, dict] = {}
    val_aggregate_subscores = getattr(result, "val_aggregate_subscores", None) or []
    if val_aggregate_subscores:
        seed_objectives = val_aggregate_subscores[0] if len(val_aggregate_subscores) > 0 else {}
        best_objectives = (
            val_aggregate_subscores[best_idx]
            if best_idx < len(val_aggregate_subscores)
            else {}
        )
        all_objectives = set(seed_objectives) | set(best_objectives)
        for obj_name in all_objectives:
            signal_improvements[obj_name] = {
                "before": seed_objectives.get(obj_name, 0.0),
                "after": best_objectives.get(obj_name, 0.0),
            }

    return OptimizationReport(
        seed_candidate=seed_config,
        best_candidate=best_candidate,
        pareto_front=pareto_front,
        total_metric_calls=getattr(result, "total_metric_calls", 0),
        total_tokens_consumed=adapter.total_tokens_consumed,
        tokens_saved_by_early_stopping=adapter.total_early_stop_tokens_saved,
        per_task_comparison=per_task_comparison,
        signal_improvements=signal_improvements,
        optimization_trace=optimization_trace,
        wall_time_seconds=wall_time,
    )
