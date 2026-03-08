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

    return OptimizationReport(
        seed_candidate=seed_config,
        best_candidate=best_candidate,
        pareto_front=pareto_front,
        total_metric_calls=getattr(result, "total_metric_calls", 0),
        total_tokens_consumed=0,
        tokens_saved_by_early_stopping=0,
        per_task_comparison={},
        signal_improvements={},
        optimization_trace=optimization_trace,
        wall_time_seconds=wall_time,
    )
