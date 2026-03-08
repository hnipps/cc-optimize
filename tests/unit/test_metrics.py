from __future__ import annotations

import pytest

from cc_optimize.evaluation.metrics import compute_coding_metrics
from cc_optimize.signals.jsonl_parser import AssistantBlock, ParsedSession, ToolCall


def _make_tool_call(
    tool_name: str,
    input_data: dict | None = None,
    is_error: bool = False,
    tool_use_id: str = "tu_1",
    idx: int = 0,
) -> ToolCall:
    return ToolCall(
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        input_data=input_data or {},
        output="ok",
        is_error=is_error,
        timestamp_index=idx,
    )


def _make_session(
    tool_calls: list[ToolCall] | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    num_blocks: int = 1,
) -> ParsedSession:
    tool_calls = tool_calls or []
    blocks = [
        AssistantBlock(text="block", tool_calls=[], index=i)
        for i in range(num_blocks)
    ]
    return ParsedSession(
        assistant_blocks=blocks,
        all_tool_calls=tool_calls,
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        raw_events=[],
    )


class TestTokens:
    def test_token_counts(self) -> None:
        session = _make_session(input_tokens=1000, output_tokens=500)
        metrics = compute_coding_metrics(session)
        assert metrics["tokens_input"] == 1000
        assert metrics["tokens_output"] == 500
        assert metrics["tokens_total"] == 1500

    def test_turn_count(self) -> None:
        session = _make_session(num_blocks=5)
        metrics = compute_coding_metrics(session)
        assert metrics["turn_count"] == 5


class TestFileEditChurn:
    def test_single_file_single_edit(self) -> None:
        calls = [
            _make_tool_call("Edit", {"file_path": "/src/main.py"}, idx=0),
        ]
        session = _make_session(tool_calls=calls)
        metrics = compute_coding_metrics(session)
        assert metrics["file_edit_churn"] == 1.0

    def test_same_file_multiple_edits(self) -> None:
        calls = [
            _make_tool_call("Edit", {"file_path": "/src/main.py"}, idx=0, tool_use_id="tu_1"),
            _make_tool_call("Write", {"file_path": "/src/main.py"}, idx=1, tool_use_id="tu_2"),
            _make_tool_call("Edit", {"file_path": "/src/main.py"}, idx=2, tool_use_id="tu_3"),
        ]
        session = _make_session(tool_calls=calls)
        metrics = compute_coding_metrics(session)
        # 3 calls / 1 file = 3.0
        assert metrics["file_edit_churn"] == 3.0

    def test_different_files(self) -> None:
        calls = [
            _make_tool_call("Edit", {"file_path": "/src/a.py"}, idx=0, tool_use_id="tu_1"),
            _make_tool_call("Write", {"file_path": "/src/b.py"}, idx=1, tool_use_id="tu_2"),
        ]
        session = _make_session(tool_calls=calls)
        metrics = compute_coding_metrics(session)
        # 2 calls / 2 files = 1.0
        assert metrics["file_edit_churn"] == 1.0

    def test_path_key_variant(self) -> None:
        calls = [
            _make_tool_call("MultiEdit", {"path": "/src/main.py"}, idx=0),
        ]
        session = _make_session(tool_calls=calls)
        metrics = compute_coding_metrics(session)
        assert metrics["file_edit_churn"] == 1.0

    def test_non_edit_tools_excluded(self) -> None:
        calls = [
            _make_tool_call("Read", {"file_path": "/src/main.py"}, idx=0),
            _make_tool_call("Bash", {"command": "ls"}, idx=1, tool_use_id="tu_2"),
        ]
        session = _make_session(tool_calls=calls)
        metrics = compute_coding_metrics(session)
        assert metrics["file_edit_churn"] == 0.0


class TestToolErrorRate:
    def test_no_errors(self) -> None:
        calls = [
            _make_tool_call("Read", idx=0, tool_use_id="tu_1"),
            _make_tool_call("Edit", idx=1, tool_use_id="tu_2"),
        ]
        session = _make_session(tool_calls=calls)
        metrics = compute_coding_metrics(session)
        assert metrics["tool_error_rate"] == 0.0

    def test_some_errors(self) -> None:
        calls = [
            _make_tool_call("Read", is_error=True, idx=0, tool_use_id="tu_1"),
            _make_tool_call("Read", is_error=False, idx=1, tool_use_id="tu_2"),
            _make_tool_call("Edit", is_error=True, idx=2, tool_use_id="tu_3"),
            _make_tool_call("Bash", is_error=False, idx=3, tool_use_id="tu_4"),
        ]
        session = _make_session(tool_calls=calls)
        metrics = compute_coding_metrics(session)
        assert metrics["tool_error_rate"] == pytest.approx(0.5)

    def test_all_errors(self) -> None:
        calls = [
            _make_tool_call("Read", is_error=True, idx=0, tool_use_id="tu_1"),
            _make_tool_call("Edit", is_error=True, idx=1, tool_use_id="tu_2"),
        ]
        session = _make_session(tool_calls=calls)
        metrics = compute_coding_metrics(session)
        assert metrics["tool_error_rate"] == 1.0


class TestEmptySession:
    def test_empty_session(self) -> None:
        session = _make_session(tool_calls=[], input_tokens=0, output_tokens=0, num_blocks=0)
        metrics = compute_coding_metrics(session)
        assert metrics["tokens_input"] == 0
        assert metrics["tokens_output"] == 0
        assert metrics["tokens_total"] == 0
        assert metrics["turn_count"] == 0
        assert metrics["file_edit_churn"] == 0.0
        assert metrics["tool_error_rate"] == 0.0
