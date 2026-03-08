from __future__ import annotations

from cc_optimize.models.benchmark import SignalBaselines
from cc_optimize.models.signals import BehavioralSignals
from cc_optimize.signals.efficiency import compute_efficiency_score
from cc_optimize.signals.jsonl_parser import ParsedSession
from cc_optimize.signals.repair import compute_repair_frequency
from cc_optimize.signals.repetition import compute_repetition
from cc_optimize.signals.tool_errors import compute_tool_error_cascade


def compute_all_signals(session: ParsedSession, baselines: SignalBaselines) -> BehavioralSignals:
    """Compute all behavioral signals for a session. Calls each signal module."""
    efficiency_score, turn_count = compute_efficiency_score(
        session, baselines.efficiency_turn_baseline
    )
    repetition_count, repetition_exact_count, repetition_max_severity = compute_repetition(session)
    tool_error_max_cascade, tool_error_total_failures, tool_error_total_calls = (
        compute_tool_error_cascade(session)
    )
    repair_frequency, repair_count = compute_repair_frequency(session)

    return BehavioralSignals(
        efficiency_score=efficiency_score,
        turn_count=turn_count,
        repetition_count=repetition_count,
        repetition_exact_count=repetition_exact_count,
        repetition_max_severity=repetition_max_severity,
        tool_error_max_cascade=tool_error_max_cascade,
        tool_error_total_failures=tool_error_total_failures,
        tool_error_total_calls=tool_error_total_calls,
        repair_frequency=repair_frequency,
        repair_count=repair_count,
    )
