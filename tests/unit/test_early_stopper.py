from __future__ import annotations

import pytest

from cc_optimize.adapter.early_stopper import should_stop
from cc_optimize.config import EarlyStopConfig
from cc_optimize.models.benchmark import SignalBaselines
from cc_optimize.signals.jsonl_parser import AssistantBlock, ParsedSession, ToolCall


def _make_session(
    num_blocks: int = 2,
    block_text: str = "unique text block number",
    tool_calls: list[ToolCall] | None = None,
    repeat_text: bool = False,
) -> ParsedSession:
    """Build a minimal ParsedSession for testing."""
    blocks: list[AssistantBlock] = []
    all_tools: list[ToolCall] = []

    for i in range(num_blocks):
        if repeat_text:
            text = "the quick brown fox jumps over the lazy dog again and again"
        else:
            text = f"{block_text} {i} with some extra words to pass threshold"
        block_tools: list[ToolCall] = []
        blocks.append(AssistantBlock(text=text, tool_calls=block_tools, index=i))

    if tool_calls is not None:
        all_tools = tool_calls

    return ParsedSession(
        assistant_blocks=blocks,
        all_tool_calls=all_tools,
        total_input_tokens=1000,
        total_output_tokens=500,
        raw_events=[],
    )


def _make_baselines(
    efficiency_turn_baseline: int = 10,
    max_looping_severity: int = 3,
    max_tool_error_cascade: int = 5,
    max_repair_frequency: float = 0.5,
) -> SignalBaselines:
    return SignalBaselines(
        efficiency_turn_baseline=efficiency_turn_baseline,
        max_looping_severity=max_looping_severity,
        max_tool_error_cascade=max_tool_error_cascade,
        max_repair_frequency=max_repair_frequency,
    )


def _make_error_tool_calls(count: int, start_index: int = 0) -> list[ToolCall]:
    """Create a cascade of error tool calls."""
    return [
        ToolCall(
            tool_use_id=f"tc_{i}",
            tool_name="Bash",
            input_data={"command": "test"},
            output="error",
            is_error=True,
            timestamp_index=start_index + i,
        )
        for i in range(count)
    ]


class TestShouldStopNoStop:
    def test_all_within_thresholds(self):
        """No stop when all signals are within thresholds."""
        session = _make_session(num_blocks=3)
        baselines = _make_baselines(efficiency_turn_baseline=10)
        config = EarlyStopConfig(
            max_looping_severity=3,
            max_tool_error_cascade=5,
            min_efficiency_score=0.1,
        )

        stop, reason = should_stop(session, baselines, config)

        assert stop is False
        assert reason == ""


class TestShouldStopRepetition:
    def test_stop_on_repetition_severity(self):
        """Stop when repetition severity reaches threshold."""
        # Build 8 consecutive identical blocks to get severity 3 (>=6 repetitions)
        session = _make_session(num_blocks=8, repeat_text=True)
        baselines = _make_baselines(efficiency_turn_baseline=20)
        config = EarlyStopConfig(
            max_looping_severity=3,
            max_tool_error_cascade=100,
            min_efficiency_score=0.0,
        )

        stop, reason = should_stop(session, baselines, config)

        assert stop is True
        assert "Repetition severity" in reason
        assert "reached threshold" in reason


class TestShouldStopToolErrorCascade:
    def test_stop_on_tool_error_cascade(self):
        """Stop when tool error cascade reaches threshold."""
        error_tools = _make_error_tool_calls(6)
        session = _make_session(num_blocks=2, tool_calls=error_tools)
        baselines = _make_baselines(efficiency_turn_baseline=10)
        config = EarlyStopConfig(
            max_looping_severity=100,
            max_tool_error_cascade=5,
            min_efficiency_score=0.0,
        )

        stop, reason = should_stop(session, baselines, config)

        assert stop is True
        assert "Tool error cascade" in reason
        assert "reached threshold" in reason


class TestShouldStopEfficiency:
    def test_stop_on_low_efficiency(self):
        """Stop when efficiency score is below threshold."""
        # Many blocks relative to baseline → low efficiency
        session = _make_session(num_blocks=50)
        baselines = _make_baselines(efficiency_turn_baseline=2)
        config = EarlyStopConfig(
            max_looping_severity=100,
            max_tool_error_cascade=100,
            min_efficiency_score=0.5,
        )

        stop, reason = should_stop(session, baselines, config)

        assert stop is True
        assert "Efficiency score" in reason
        assert "below threshold" in reason


class TestShouldStopPriority:
    def test_first_condition_triggers(self):
        """When multiple thresholds exceeded, repetition (checked first) triggers."""
        # Both repetition and tool error cascade exceed thresholds
        error_tools = _make_error_tool_calls(10)
        session = _make_session(num_blocks=8, repeat_text=True, tool_calls=error_tools)
        baselines = _make_baselines(efficiency_turn_baseline=20)
        config = EarlyStopConfig(
            max_looping_severity=3,
            max_tool_error_cascade=5,
            min_efficiency_score=0.0,
        )

        stop, reason = should_stop(session, baselines, config)

        assert stop is True
        # Repetition is checked first
        assert "Repetition severity" in reason
