from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cc_optimize.signals.jsonl_parser import parse, ParsedSession

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestSampleSession:
    def test_assistant_block_count(self) -> None:
        session = parse(FIXTURES_DIR / "sample_session.jsonl")
        assert len(session.assistant_blocks) == 2

    def test_tool_call_count(self) -> None:
        session = parse(FIXTURES_DIR / "sample_session.jsonl")
        assert len(session.all_tool_calls) == 2

    def test_token_counts(self) -> None:
        session = parse(FIXTURES_DIR / "sample_session.jsonl")
        assert session.total_input_tokens == 1500
        assert session.total_output_tokens == 350

    def test_tool_call_names(self) -> None:
        session = parse(FIXTURES_DIR / "sample_session.jsonl")
        names = [tc.tool_name for tc in session.all_tool_calls]
        assert names == ["Read", "Edit"]

    def test_tool_calls_successful(self) -> None:
        session = parse(FIXTURES_DIR / "sample_session.jsonl")
        assert all(not tc.is_error for tc in session.all_tool_calls)

    def test_tool_output_populated(self) -> None:
        session = parse(FIXTURES_DIR / "sample_session.jsonl")
        assert session.all_tool_calls[0].output == "def main():\n    print('hello')"
        assert session.all_tool_calls[1].output == "File updated successfully."


class TestLoopingSession:
    def test_seven_assistant_blocks(self) -> None:
        session = parse(FIXTURES_DIR / "looping_session.jsonl")
        assert len(session.assistant_blocks) == 7

    def test_all_blocks_have_similar_text(self) -> None:
        session = parse(FIXTURES_DIR / "looping_session.jsonl")
        texts = [b.text for b in session.assistant_blocks]
        assert len(set(texts)) == 1  # all identical

    def test_no_tool_calls(self) -> None:
        session = parse(FIXTURES_DIR / "looping_session.jsonl")
        assert len(session.all_tool_calls) == 0


class TestErrorCascade:
    def test_error_detection(self) -> None:
        session = parse(FIXTURES_DIR / "error_cascade.jsonl")
        errors = [tc.is_error for tc in session.all_tool_calls]
        assert errors == [True, True, True, False]

    def test_four_tool_calls(self) -> None:
        session = parse(FIXTURES_DIR / "error_cascade.jsonl")
        assert len(session.all_tool_calls) == 4

    def test_four_assistant_blocks(self) -> None:
        session = parse(FIXTURES_DIR / "error_cascade.jsonl")
        assert len(session.assistant_blocks) == 4

    def test_token_counts(self) -> None:
        session = parse(FIXTURES_DIR / "error_cascade.jsonl")
        assert session.total_input_tokens == 3200
        assert session.total_output_tokens == 800


class TestMalformedInput:
    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        jsonl_file = tmp_path / "malformed.jsonl"
        jsonl_file.write_text(
            '{"type": "assistant", "content": [{"type": "text", "text": "hello"}]}\n'
            "this is not valid json\n"
            '{"type": "result", "usage": {"input_tokens": 10, "output_tokens": 5}}\n'
        )
        session = parse(jsonl_file)
        assert len(session.assistant_blocks) == 1
        assert session.total_input_tokens == 10

    def test_empty_file(self, tmp_path: Path) -> None:
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")
        session = parse(jsonl_file)
        assert len(session.assistant_blocks) == 0
        assert len(session.all_tool_calls) == 0
        assert session.total_input_tokens == 0
        assert session.total_output_tokens == 0

    def test_missing_result_event(self, tmp_path: Path) -> None:
        jsonl_file = tmp_path / "no_result.jsonl"
        jsonl_file.write_text(
            '{"type": "assistant", "content": [{"type": "text", "text": "hello"}]}\n'
        )
        session = parse(jsonl_file)
        assert session.total_input_tokens == 0
        assert session.total_output_tokens == 0

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        jsonl_file = tmp_path / "blanks.jsonl"
        jsonl_file.write_text(
            "\n"
            '{"type": "assistant", "content": [{"type": "text", "text": "hi"}]}\n'
            "\n"
            '{"type": "result", "usage": {"input_tokens": 1, "output_tokens": 2}}\n'
            "\n"
        )
        session = parse(jsonl_file)
        assert len(session.assistant_blocks) == 1
        assert session.total_input_tokens == 1

    def test_unknown_event_types_skipped(self, tmp_path: Path) -> None:
        jsonl_file = tmp_path / "unknown.jsonl"
        jsonl_file.write_text(
            '{"type": "debug", "data": "should be ignored"}\n'
            '{"type": "assistant", "content": [{"type": "text", "text": "ok"}]}\n'
            '{"type": "result", "usage": {"input_tokens": 5, "output_tokens": 3}}\n'
        )
        session = parse(jsonl_file)
        assert len(session.assistant_blocks) == 1
