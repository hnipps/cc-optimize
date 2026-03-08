from __future__ import annotations

from cc_optimize.signals.jsonl_parser import ParsedSession, ToolCall
from cc_optimize.signals.tool_errors import compute_tool_error_cascade


def make_session(assistant_blocks=None, tool_calls=None):
    return ParsedSession(
        assistant_blocks=assistant_blocks or [],
        all_tool_calls=tool_calls or [],
        total_input_tokens=0,
        total_output_tokens=0,
        raw_events=[],
    )


def _tc(index: int, is_error: bool) -> ToolCall:
    return ToolCall(
        tool_use_id=f"tc_{index}",
        tool_name="Bash",
        input_data={},
        output="out",
        is_error=is_error,
        timestamp_index=index,
    )


class TestComputeToolErrorCascade:
    def test_all_success(self):
        calls = [_tc(0, False), _tc(1, False), _tc(2, False)]
        session = make_session(tool_calls=calls)
        max_cascade, failures, total = compute_tool_error_cascade(session)
        assert max_cascade == 0
        assert failures == 0
        assert total == 3

    def test_single_failure_then_success(self):
        calls = [_tc(0, True), _tc(1, False)]
        session = make_session(tool_calls=calls)
        max_cascade, failures, total = compute_tool_error_cascade(session)
        assert max_cascade == 1
        assert failures == 1
        assert total == 2

    def test_five_failures_then_success(self):
        calls = [_tc(i, True) for i in range(5)] + [_tc(5, False)]
        session = make_session(tool_calls=calls)
        max_cascade, failures, total = compute_tool_error_cascade(session)
        assert max_cascade == 5
        assert failures == 5
        assert total == 6

    def test_interleaved(self):
        # S, F, F, S, F, S
        patterns = [False, True, True, False, True, False]
        calls = [_tc(i, err) for i, err in enumerate(patterns)]
        session = make_session(tool_calls=calls)
        max_cascade, failures, total = compute_tool_error_cascade(session)
        assert max_cascade == 2
        assert failures == 3
        assert total == 6

    def test_trailing_failures(self):
        # F, F, F (no trailing success)
        calls = [_tc(0, True), _tc(1, True), _tc(2, True)]
        session = make_session(tool_calls=calls)
        max_cascade, failures, total = compute_tool_error_cascade(session)
        assert max_cascade == 3
        assert failures == 3
        assert total == 3

    def test_empty(self):
        session = make_session(tool_calls=[])
        max_cascade, failures, total = compute_tool_error_cascade(session)
        assert max_cascade == 0
        assert failures == 0
        assert total == 0
