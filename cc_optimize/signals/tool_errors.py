from __future__ import annotations

from cc_optimize.signals.jsonl_parser import ParsedSession


def compute_tool_error_cascade(session: ParsedSession) -> tuple[int, int, int]:
    """Returns (max_cascade, total_failures, total_calls).
    Walk all_tool_calls in timestamp_index order.
    On failure: current_cascade += 1
    On success: record current_cascade, reset to 0.
    At end: record final current_cascade.
    max_cascade = max of all recorded cascade lengths.
    """
    calls = sorted(session.all_tool_calls, key=lambda tc: tc.timestamp_index)
    total_calls = len(calls)
    total_failures = sum(1 for tc in calls if tc.is_error)

    if not calls:
        return 0, 0, 0

    current_cascade = 0
    max_cascade = 0

    for tc in calls:
        if tc.is_error:
            current_cascade += 1
        else:
            max_cascade = max(max_cascade, current_cascade)
            current_cascade = 0

    # Record final cascade (handles trailing failures)
    max_cascade = max(max_cascade, current_cascade)

    return max_cascade, total_failures, total_calls
