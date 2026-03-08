from __future__ import annotations

from cc_optimize.signals.jsonl_parser import ParsedSession


def compute_coding_metrics(session: ParsedSession) -> dict:
    """
    Returns {
        "tokens_input": int,
        "tokens_output": int,
        "tokens_total": int,
        "turn_count": int,
        "file_edit_churn": float,
        "tool_error_rate": float,
    }

    file_edit_churn: count tool calls with tool_name in ("Write", "Edit", "MultiEdit"),
    extract target file path from input_data (key "file_path" or "path"),
    churn = total_write_edit_calls / distinct_files_targeted. 0.0 if none.

    tool_error_rate: total_failures / total_tool_calls. 0.0 if no calls.
    """
    tokens_input = session.total_input_tokens
    tokens_output = session.total_output_tokens
    tokens_total = tokens_input + tokens_output
    turn_count = len(session.assistant_blocks)

    # File edit churn
    edit_tool_names = {"Write", "Edit", "MultiEdit"}
    edit_calls = 0
    targeted_files: set[str] = set()

    for tc in session.all_tool_calls:
        if tc.tool_name in edit_tool_names:
            edit_calls += 1
            file_path = tc.input_data.get("file_path") or tc.input_data.get("path")
            if file_path:
                targeted_files.add(file_path)

    if edit_calls > 0 and targeted_files:
        file_edit_churn = edit_calls / len(targeted_files)
    else:
        file_edit_churn = 0.0

    # Tool error rate
    total_calls = len(session.all_tool_calls)
    total_failures = sum(1 for tc in session.all_tool_calls if tc.is_error)

    if total_calls > 0:
        tool_error_rate = total_failures / total_calls
    else:
        tool_error_rate = 0.0

    return {
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "tokens_total": tokens_total,
        "turn_count": turn_count,
        "file_edit_churn": file_edit_churn,
        "tool_error_rate": tool_error_rate,
    }
