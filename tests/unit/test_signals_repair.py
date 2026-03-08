from __future__ import annotations

import pytest

from cc_optimize.signals.jsonl_parser import ParsedSession, ToolCall
from cc_optimize.signals.repair import compute_repair_frequency


def make_session(assistant_blocks=None, tool_calls=None):
    return ParsedSession(
        assistant_blocks=assistant_blocks or [],
        all_tool_calls=tool_calls or [],
        total_input_tokens=0,
        total_output_tokens=0,
        raw_events=[],
    )


def _tc(index: int, tool_name: str, is_error: bool) -> ToolCall:
    return ToolCall(
        tool_use_id=f"tc_{index}",
        tool_name=tool_name,
        input_data={},
        output="out",
        is_error=is_error,
        timestamp_index=index,
    )


class TestComputeRepairFrequency:
    def test_bash_fail_then_bash_success(self):
        calls = [_tc(0, "Bash", True), _tc(1, "Bash", False)]
        session = make_session(tool_calls=calls)
        freq, count = compute_repair_frequency(session)
        assert count == 1
        assert freq == pytest.approx(0.5)

    def test_bash_fail_then_write_success(self):
        calls = [_tc(0, "Bash", True), _tc(1, "Write", False)]
        session = make_session(tool_calls=calls)
        freq, count = compute_repair_frequency(session)
        assert count == 0
        assert freq == 0.0

    def test_double_repair(self):
        # Bash(fail), Bash(fail), Bash(success)
        calls = [_tc(0, "Bash", True), _tc(1, "Bash", True), _tc(2, "Bash", False)]
        session = make_session(tool_calls=calls)
        freq, count = compute_repair_frequency(session)
        assert count == 2
        assert freq == pytest.approx(2.0 / 3.0)

    def test_no_repairs_all_success(self):
        calls = [_tc(0, "Bash", False), _tc(1, "Bash", False)]
        session = make_session(tool_calls=calls)
        freq, count = compute_repair_frequency(session)
        assert count == 0
        assert freq == 0.0

    def test_empty(self):
        session = make_session(tool_calls=[])
        freq, count = compute_repair_frequency(session)
        assert count == 0
        assert freq == 0.0
