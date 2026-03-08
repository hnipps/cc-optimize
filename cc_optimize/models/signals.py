from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BehavioralSignals:
    efficiency_score: float
    turn_count: int
    repetition_count: int
    repetition_exact_count: int
    repetition_max_severity: int
    tool_error_max_cascade: int
    tool_error_total_failures: int
    tool_error_total_calls: int
    repair_frequency: float
    repair_count: int
    early_stopped: bool = False
    early_stop_reason: str = ""
