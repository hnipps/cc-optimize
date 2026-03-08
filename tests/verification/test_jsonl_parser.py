"""Verify JSONL parser correctness against spec Section 4.1.

BUG FIXED: The parser originally read content from event.get("content", [])
but the spec documents content nested under event.message.content. This was
fixed in jsonl_parser.py to use event.get("message", {}).get("content", []).
"""
from __future__ import annotations

import json
from pathlib import Path

from cc_optimize.signals.jsonl_parser import parse


def _write_jsonl(tmp_path: Path, events: list) -> Path:
    """Write a list of dicts/strings as a JSONL file."""
    p = tmp_path / "session.jsonl"
    lines = []
    for e in events:
        if isinstance(e, str):
            lines.append(e)
        else:
            lines.append(json.dumps(e))
    p.write_text("\n".join(lines) + "\n")
    return p


class TestAssistantTextMessage:
    def test_single_text_block(self, tmp_path):
        events = [
            {"type": "system", "subtype": "init", "session_id": "s1"},
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Hello world"}
            ]}},
            {"type": "result", "result": "done", "usage": {"input_tokens": 100, "output_tokens": 50}},
        ]
        session = parse(_write_jsonl(tmp_path, events))
        assert len(session.assistant_blocks) == 1
        assert session.assistant_blocks[0].text == "Hello world"
        assert session.assistant_blocks[0].tool_calls == []
        assert session.total_input_tokens == 100
        assert session.total_output_tokens == 50


class TestAssistantToolUse:
    def test_tool_use_message(self, tmp_path):
        events = [
            {"type": "system", "subtype": "init", "session_id": "s1"},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "tc_1", "name": "Bash", "input": {"command": "ls"}}
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc_1", "content": "file1.py\nfile2.py", "is_error": False}
            ]}},
            {"type": "result", "result": "done", "usage": {"input_tokens": 200, "output_tokens": 100}},
        ]
        session = parse(_write_jsonl(tmp_path, events))
        assert len(session.assistant_blocks) == 1
        assert len(session.all_tool_calls) == 1
        tc = session.all_tool_calls[0]
        assert tc.tool_name == "Bash"
        assert tc.tool_use_id == "tc_1"
        assert tc.is_error is False
        assert tc.output == "file1.py\nfile2.py"


class TestMixedTextAndToolUse:
    def test_mixed_content(self, tmp_path):
        events = [
            {"type": "system", "subtype": "init", "session_id": "s1"},
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "I'll run that command..."},
                {"type": "tool_use", "id": "tc_1", "name": "Bash", "input": {"command": "go test ./..."}}
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc_1", "content": "PASS", "is_error": False}
            ]}},
            {"type": "result", "result": "done", "usage": {"input_tokens": 300, "output_tokens": 150}},
        ]
        session = parse(_write_jsonl(tmp_path, events))
        assert len(session.assistant_blocks) == 1
        block = session.assistant_blocks[0]
        assert "I'll run that command" in block.text
        assert len(block.tool_calls) == 1
        assert block.tool_calls[0].tool_name == "Bash"


class TestToolResultError:
    def test_error_tool_result(self, tmp_path):
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "tc_1", "name": "Bash", "input": {"command": "nonexistent"}}
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc_1", "content": "command not found", "is_error": True}
            ]}},
            {"type": "result", "result": "done", "usage": {"input_tokens": 50, "output_tokens": 25}},
        ]
        session = parse(_write_jsonl(tmp_path, events))
        assert len(session.all_tool_calls) == 1
        assert session.all_tool_calls[0].is_error is True
        assert session.all_tool_calls[0].output == "command not found"


class TestFinalResultWithUsage:
    def test_token_counts(self, tmp_path):
        events = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
            {"type": "result", "result": "task complete", "session_id": "s1",
             "usage": {"input_tokens": 1234, "output_tokens": 567}},
        ]
        session = parse(_write_jsonl(tmp_path, events))
        assert session.total_input_tokens == 1234
        assert session.total_output_tokens == 567


class TestMissingFinalResult:
    """Known CLI bug: final result event may be missing. Parser should not crash."""

    def test_no_result_event(self, tmp_path):
        events = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "working on it"}]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "tc_1", "name": "Write", "input": {"file_path": "a.py"}}
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc_1", "content": "ok", "is_error": False}
            ]}},
            # No result event!
        ]
        session = parse(_write_jsonl(tmp_path, events))
        # Should not crash; tokens default to 0 (or fallback aggregation)
        assert session.total_input_tokens >= 0
        assert session.total_output_tokens >= 0
        assert len(session.assistant_blocks) >= 1


class TestMalformedJsonLine:
    def test_malformed_line_skipped(self, tmp_path):
        events = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "hello"}]}},
            "this is not valid json {{{",
            {"type": "result", "result": "done", "usage": {"input_tokens": 10, "output_tokens": 5}},
        ]
        session = parse(_write_jsonl(tmp_path, events))
        # Malformed line skipped, rest parsed fine
        assert len(session.assistant_blocks) == 1
        assert session.total_input_tokens == 10


class TestUnknownEventType:
    def test_stream_event_skipped(self, tmp_path):
        events = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "hello"}]}},
            {"type": "stream_event", "data": "some streaming data"},
            {"type": "result", "result": "done", "usage": {"input_tokens": 10, "output_tokens": 5}},
        ]
        session = parse(_write_jsonl(tmp_path, events))
        assert len(session.assistant_blocks) == 1
        # stream_event should be in raw_events but filtered from processing
        assert any(e.get("type") == "stream_event" for e in session.raw_events)


class TestMultipleTurns:
    """Verify multiple assistant/user turns create correct blocks."""

    def test_two_assistant_turns(self, tmp_path):
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "First turn"},
                {"type": "tool_use", "id": "tc_1", "name": "Read", "input": {"path": "a.py"}}
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc_1", "content": "file content", "is_error": False}
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Second turn"},
                {"type": "tool_use", "id": "tc_2", "name": "Edit", "input": {"file_path": "a.py"}}
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc_2", "content": "edited", "is_error": False}
            ]}},
            {"type": "result", "result": "done", "usage": {"input_tokens": 500, "output_tokens": 200}},
        ]
        session = parse(_write_jsonl(tmp_path, events))
        assert len(session.assistant_blocks) == 2
        assert session.assistant_blocks[0].text == "First turn"
        assert session.assistant_blocks[1].text == "Second turn"
        assert len(session.all_tool_calls) == 2
        assert session.all_tool_calls[0].tool_name == "Read"
        assert session.all_tool_calls[1].tool_name == "Edit"
        assert session.all_tool_calls[0].is_error is False
        assert session.all_tool_calls[1].is_error is False


class TestEmptyFile:
    def test_empty_jsonl(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        session = parse(p)
        assert session.assistant_blocks == []
        assert session.all_tool_calls == []
        assert session.total_input_tokens == 0
        assert session.total_output_tokens == 0
