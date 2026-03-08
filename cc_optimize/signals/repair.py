from __future__ import annotations

from cc_optimize.signals.jsonl_parser import ParsedSession


def compute_repair_frequency(session: ParsedSession) -> tuple[float, int]:
    """Returns (repair_frequency, repair_count).
    A tool call at index i is a "repair" if:
    1. i > 0
    2. previous tool call has same tool_name
    3. previous tool call is_error == True
    repair_frequency = repair_count / len(all_tool_calls) (0.0 if empty)
    """
    calls = sorted(session.all_tool_calls, key=lambda tc: tc.timestamp_index)
    if not calls:
        return 0.0, 0

    repair_count = 0
    for i in range(1, len(calls)):
        prev = calls[i - 1]
        curr = calls[i]
        if curr.tool_name == prev.tool_name and prev.is_error:
            repair_count += 1

    repair_frequency = repair_count / len(calls)
    return repair_frequency, repair_count
