"""Verify metrics computation from spec Section 4.8.

- file_edit_churn = total write/edit calls / distinct files
- tool_error_rate = failures / total calls
- Edge cases: no tool calls
"""
from __future__ import annotations

import pytest

from cc_optimize.evaluation.metrics import compute_coding_metrics
from cc_optimize.signals.jsonl_parser import AssistantBlock, ParsedSession, ToolCall


def _tc(index: int, tool_name: str, file_path: str | None = None, is_error: bool = False) -> ToolCall:
    input_data = {}
    if file_path:
        input_data["file_path"] = file_path
    return ToolCall(
        tool_use_id=f"tc_{index}",
        tool_name=tool_name,
        input_data=input_data,
        output=None,
        is_error=is_error,
        timestamp_index=index,
    )


def _make_session(tool_calls: list[ToolCall], input_tokens: int = 500, output_tokens: int = 200) -> ParsedSession:
    return ParsedSession(
        assistant_blocks=[AssistantBlock(text="x", tool_calls=[], index=0)],
        all_tool_calls=tool_calls,
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        raw_events=[],
    )


class TestFileEditChurn:
    def test_spec_example(self):
        """6 Write/Edit calls to 3 distinct files -> churn = 2.0"""
        calls = [
            _tc(0, "Write", "/a.py"),
            _tc(1, "Edit", "/a.py"),
            _tc(2, "Write", "/b.py"),
            _tc(3, "Edit", "/b.py"),
            _tc(4, "Write", "/c.py"),
            _tc(5, "Edit", "/c.py"),
        ]
        session = _make_session(calls)
        metrics = compute_coding_metrics(session)
        assert metrics["file_edit_churn"] == pytest.approx(2.0)

    def test_single_file_many_edits(self):
        """4 edits to 1 file -> churn = 4.0"""
        calls = [_tc(i, "Edit", "/x.py") for i in range(4)]
        session = _make_session(calls)
        metrics = compute_coding_metrics(session)
        assert metrics["file_edit_churn"] == pytest.approx(4.0)

    def test_many_files_one_edit_each(self):
        """5 files with 1 edit each -> churn = 1.0"""
        calls = [_tc(i, "Write", f"/file{i}.py") for i in range(5)]
        session = _make_session(calls)
        metrics = compute_coding_metrics(session)
        assert metrics["file_edit_churn"] == pytest.approx(1.0)

    def test_no_edit_calls(self):
        """No Write/Edit calls -> churn = 0.0"""
        calls = [_tc(0, "Bash"), _tc(1, "Read")]
        session = _make_session(calls)
        metrics = compute_coding_metrics(session)
        assert metrics["file_edit_churn"] == pytest.approx(0.0)

    def test_multi_edit_counted(self):
        """MultiEdit tool calls are counted in churn."""
        calls = [
            _tc(0, "MultiEdit", "/a.py"),
            _tc(1, "MultiEdit", "/a.py"),
            _tc(2, "Write", "/b.py"),
        ]
        session = _make_session(calls)
        metrics = compute_coding_metrics(session)
        # 3 edit calls / 2 files = 1.5
        assert metrics["file_edit_churn"] == pytest.approx(1.5)


class TestToolErrorRate:
    def test_spec_example(self):
        """2 failures out of 10 -> 0.2"""
        calls = [_tc(i, "Bash", is_error=(i < 2)) for i in range(10)]
        session = _make_session(calls)
        metrics = compute_coding_metrics(session)
        assert metrics["tool_error_rate"] == pytest.approx(0.2)

    def test_no_failures(self):
        calls = [_tc(i, "Bash") for i in range(5)]
        session = _make_session(calls)
        metrics = compute_coding_metrics(session)
        assert metrics["tool_error_rate"] == pytest.approx(0.0)

    def test_all_failures(self):
        calls = [_tc(i, "Bash", is_error=True) for i in range(3)]
        session = _make_session(calls)
        metrics = compute_coding_metrics(session)
        assert metrics["tool_error_rate"] == pytest.approx(1.0)


class TestEdgeCases:
    def test_no_tool_calls(self):
        """No tool calls -> churn=0.0, error_rate=0.0"""
        session = _make_session([])
        metrics = compute_coding_metrics(session)
        assert metrics["file_edit_churn"] == pytest.approx(0.0)
        assert metrics["tool_error_rate"] == pytest.approx(0.0)

    def test_token_counts(self):
        """Token counts come from ParsedSession."""
        session = _make_session([], input_tokens=1234, output_tokens=567)
        metrics = compute_coding_metrics(session)
        assert metrics["tokens_input"] == 1234
        assert metrics["tokens_output"] == 567
        assert metrics["tokens_total"] == 1234 + 567

    def test_turn_count(self):
        """Turn count = number of assistant blocks."""
        session = ParsedSession(
            assistant_blocks=[
                AssistantBlock(text="a", tool_calls=[], index=0),
                AssistantBlock(text="b", tool_calls=[], index=1),
                AssistantBlock(text="c", tool_calls=[], index=2),
            ],
            all_tool_calls=[],
            total_input_tokens=0,
            total_output_tokens=0,
            raw_events=[],
        )
        metrics = compute_coding_metrics(session)
        assert metrics["turn_count"] == 3
