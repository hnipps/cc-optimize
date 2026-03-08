"""Verify compute_all_signals() is a pure function (spec invariant).

The spec states that the same ParsedSession + same SignalBaselines must produce
identical BehavioralSignals every time. We run it 100 times and assert equality.
"""
from __future__ import annotations

from dataclasses import asdict

from cc_optimize.models.benchmark import SignalBaselines
from cc_optimize.signals.compute import compute_all_signals
from cc_optimize.signals.jsonl_parser import AssistantBlock, ParsedSession, ToolCall


def _build_session() -> ParsedSession:
    """A non-trivial session with text, tool calls, errors, and repairs."""
    tool_calls = [
        ToolCall("tc_0", "Bash", {"command": "ls"}, "output", False, 0),
        ToolCall("tc_1", "Bash", {"command": "bad"}, "error", True, 1),
        ToolCall("tc_2", "Bash", {"command": "bad"}, "error", True, 2),
        ToolCall("tc_3", "Write", {"file_path": "/a.py"}, None, False, 3),
        ToolCall("tc_4", "Edit", {"file_path": "/a.py"}, None, False, 4),
        ToolCall("tc_5", "Bash", {"command": "test"}, "ok", False, 5),
    ]
    long_text = "the quick brown fox jumps over the lazy dog and does many things today"
    blocks = [
        AssistantBlock(text=long_text, tool_calls=[tool_calls[0]], index=0),
        AssistantBlock(text=long_text, tool_calls=[tool_calls[1]], index=1),
        AssistantBlock(text="something completely different with enough words here", tool_calls=[tool_calls[2], tool_calls[3]], index=2),
        AssistantBlock(text=long_text, tool_calls=[tool_calls[4]], index=3),
        AssistantBlock(text="wrapping up the final work output here with words", tool_calls=[tool_calls[5]], index=4),
    ]
    return ParsedSession(
        assistant_blocks=blocks,
        all_tool_calls=tool_calls,
        total_input_tokens=5000,
        total_output_tokens=2000,
        raw_events=[],
    )


def test_determinism_100_runs():
    session = _build_session()
    baselines = SignalBaselines(
        efficiency_turn_baseline=8,
        max_looping_severity=0,
        max_tool_error_cascade=2,
        max_repair_frequency=0.15,
    )

    first_result = compute_all_signals(session, baselines)
    first_dict = asdict(first_result)

    for i in range(99):
        result = compute_all_signals(session, baselines)
        assert asdict(result) == first_dict, f"Run {i+2} produced different result"
