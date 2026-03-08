from __future__ import annotations

from cc_optimize.config import EarlyStopConfig
from cc_optimize.models.benchmark import SignalBaselines
from cc_optimize.signals.compute import compute_all_signals
from cc_optimize.signals.jsonl_parser import ParsedSession


def should_stop(
    partial_session: ParsedSession,
    baselines: SignalBaselines,
    config: EarlyStopConfig,
) -> tuple[bool, str]:
    """
    Compute signals on the partial session and check thresholds.

    Stop conditions (any one triggers):
    1. repetition_max_severity >= config.max_looping_severity
       -> reason: "Repetition severity {N} reached threshold {M}"
    2. tool_error_max_cascade >= config.max_tool_error_cascade
       -> reason: "Tool error cascade of {N} reached threshold {M}"
    3. efficiency_score < config.min_efficiency_score
       -> reason: "Efficiency score {N:.2f} below threshold {M:.2f}"

    Returns (False, "") if no stop condition is met.
    """
    signals = compute_all_signals(partial_session, baselines)

    if signals.repetition_max_severity >= config.max_looping_severity:
        return (
            True,
            f"Repetition severity {signals.repetition_max_severity} reached threshold {config.max_looping_severity}",
        )

    if signals.tool_error_max_cascade >= config.max_tool_error_cascade:
        return (
            True,
            f"Tool error cascade of {signals.tool_error_max_cascade} reached threshold {config.max_tool_error_cascade}",
        )

    if signals.efficiency_score < config.min_efficiency_score:
        return (
            True,
            f"Efficiency score {signals.efficiency_score:.2f} below threshold {config.min_efficiency_score:.2f}",
        )

    return (False, "")
