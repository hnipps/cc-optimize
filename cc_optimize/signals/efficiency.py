from __future__ import annotations

from cc_optimize.signals.jsonl_parser import ParsedSession


def compute_efficiency_score(session: ParsedSession, baseline_turns: int) -> tuple[float, int]:
    """
    Returns (efficiency_score, turn_count).
    efficiency_score = 1 / (1 + 0.3 * max(0, turn_count - baseline_turns))
    turn_count = len(session.assistant_blocks)
    """
    turn_count = len(session.assistant_blocks)
    excess = max(0, turn_count - baseline_turns)
    efficiency_score = 1.0 / (1.0 + 0.3 * excess)
    return efficiency_score, turn_count
