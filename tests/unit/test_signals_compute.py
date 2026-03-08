from __future__ import annotations

import pytest

from cc_optimize.models.benchmark import SignalBaselines
from cc_optimize.models.signals import BehavioralSignals
from cc_optimize.signals.compute import compute_all_signals
from cc_optimize.signals.jsonl_parser import AssistantBlock, ParsedSession, ToolCall


def make_session(assistant_blocks=None, tool_calls=None):
    return ParsedSession(
        assistant_blocks=assistant_blocks or [],
        all_tool_calls=tool_calls or [],
        total_input_tokens=0,
        total_output_tokens=0,
        raw_events=[],
    )


DEFAULT_BASELINES = SignalBaselines(
    efficiency_turn_baseline=10,
    max_looping_severity=3,
    max_tool_error_cascade=5,
    max_repair_frequency=0.5,
)


class TestComputeAllSignals:
    def test_empty_session(self):
        session = make_session()
        signals = compute_all_signals(session, DEFAULT_BASELINES)
        assert isinstance(signals, BehavioralSignals)
        assert signals.efficiency_score == 1.0
        assert signals.turn_count == 0
        assert signals.repetition_count == 0
        assert signals.repetition_exact_count == 0
        assert signals.repetition_max_severity == 0
        assert signals.tool_error_max_cascade == 0
        assert signals.tool_error_total_failures == 0
        assert signals.tool_error_total_calls == 0
        assert signals.repair_frequency == 0.0
        assert signals.repair_count == 0

    def test_session_with_blocks_and_tools(self):
        long_text = "the quick brown fox jumps over the lazy dog near the river bank"
        blocks = [
            AssistantBlock(text=long_text, tool_calls=[], index=0),
            AssistantBlock(text=long_text, tool_calls=[], index=1),
        ]
        calls = [
            ToolCall(
                tool_use_id="tc_0",
                tool_name="Bash",
                input_data={},
                output="out",
                is_error=True,
                timestamp_index=0,
            ),
            ToolCall(
                tool_use_id="tc_1",
                tool_name="Bash",
                input_data={},
                output="out",
                is_error=False,
                timestamp_index=1,
            ),
        ]
        session = make_session(assistant_blocks=blocks, tool_calls=calls)
        baselines = SignalBaselines(
            efficiency_turn_baseline=5,
            max_looping_severity=3,
            max_tool_error_cascade=5,
            max_repair_frequency=0.5,
        )
        signals = compute_all_signals(session, baselines)
        assert signals.turn_count == 2
        # 2 turns, baseline 5 => no excess => score 1.0
        assert signals.efficiency_score == 1.0
        # Two identical long blocks => 1 repetition
        assert signals.repetition_count == 1
        assert signals.repetition_exact_count == 1
        assert signals.repetition_max_severity == 1
        # One failure then success => cascade=1
        assert signals.tool_error_max_cascade == 1
        assert signals.tool_error_total_failures == 1
        assert signals.tool_error_total_calls == 2
        # Bash fail then Bash success => 1 repair
        assert signals.repair_count == 1
        assert signals.repair_frequency == pytest.approx(0.5)

