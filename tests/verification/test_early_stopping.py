"""Verify early stopper thresholds from spec Section 4.6.

Tests should_stop() against each of the three conditions:
1. repetition_max_severity >= threshold
2. tool_error_max_cascade >= threshold
3. efficiency_score < threshold
"""
from __future__ import annotations

from cc_optimize.adapter.early_stopper import should_stop
from cc_optimize.config import EarlyStopConfig
from cc_optimize.models.benchmark import SignalBaselines
from cc_optimize.signals.jsonl_parser import AssistantBlock, ParsedSession, ToolCall


def _baselines() -> SignalBaselines:
    return SignalBaselines(
        efficiency_turn_baseline=10,
        max_looping_severity=0,
        max_tool_error_cascade=2,
        max_repair_frequency=0.15,
    )


def _config() -> EarlyStopConfig:
    return EarlyStopConfig(
        max_looping_severity=3,
        max_tool_error_cascade=5,
        min_efficiency_score=0.2,
    )


def _make_session(
    num_blocks: int = 5,
    block_text: str = "doing work with enough words for the repetition check here",
    tool_calls: list[ToolCall] | None = None,
) -> ParsedSession:
    blocks = [
        AssistantBlock(text=block_text, tool_calls=[], index=i)
        for i in range(num_blocks)
    ]
    return ParsedSession(
        assistant_blocks=blocks,
        all_tool_calls=tool_calls or [],
        total_input_tokens=0,
        total_output_tokens=0,
        raw_events=[],
    )


def _tc(index: int, is_error: bool = False) -> ToolCall:
    return ToolCall(f"tc_{index}", "Bash", {}, None, is_error, index)


class TestRepetitionThreshold:
    def test_severity_at_threshold_stops(self):
        """repetition_max_severity >= 3 -> stop"""
        # 7 identical blocks -> 6 pairs -> severity 3
        long_text = "the quick brown fox jumps over the lazy dog and does many things"
        session = _make_session(num_blocks=7, block_text=long_text)
        stop, reason = should_stop(session, _baselines(), _config())
        assert stop is True
        assert "Repetition severity" in reason
        assert "threshold" in reason

    def test_severity_below_threshold_continues(self):
        # All different blocks -> severity 0
        blocks = [
            AssistantBlock(text=f"unique text block number {i} with enough words here today", tool_calls=[], index=i)
            for i in range(5)
        ]
        session = ParsedSession(
            assistant_blocks=blocks,
            all_tool_calls=[],
            total_input_tokens=0,
            total_output_tokens=0,
            raw_events=[],
        )
        stop, reason = should_stop(session, _baselines(), _config())
        assert stop is False
        assert reason == ""


class TestToolErrorCascadeThreshold:
    def test_cascade_at_threshold_stops(self):
        """tool_error_max_cascade >= 5 -> stop"""
        calls = [_tc(i, is_error=True) for i in range(5)]
        session = _make_session(num_blocks=1, block_text="x", tool_calls=calls)
        stop, reason = should_stop(session, _baselines(), _config())
        assert stop is True
        assert "Tool error cascade" in reason
        assert "threshold" in reason

    def test_cascade_below_threshold_continues(self):
        """cascade=4 < threshold=5 -> no stop"""
        calls = [_tc(i, is_error=True) for i in range(4)]
        calls.append(_tc(4, is_error=False))
        session = _make_session(num_blocks=1, block_text="x", tool_calls=calls)
        stop, reason = should_stop(session, _baselines(), _config())
        assert stop is False


class TestEfficiencyThreshold:
    def test_efficiency_below_threshold_stops(self):
        """efficiency_score < 0.2 -> stop"""
        # With baseline=10, we need many turns to get efficiency < 0.2
        # 1/(1+0.3*excess) < 0.2 -> 1+0.3*excess > 5 -> excess > 13.33 -> excess=14 -> turns=24
        session = _make_session(num_blocks=24, block_text="x")
        stop, reason = should_stop(session, _baselines(), _config())
        assert stop is True
        assert "Efficiency score" in reason
        assert "below threshold" in reason

    def test_efficiency_above_threshold_continues(self):
        """efficiency at baseline -> 1.0 -> no stop"""
        session = _make_session(num_blocks=10, block_text="x")
        stop, reason = should_stop(session, _baselines(), _config())
        assert stop is False


class TestBoundaryConditions:
    def test_severity_exactly_at_threshold(self):
        """Exactly at threshold (>=) should stop."""
        config = EarlyStopConfig(max_looping_severity=3, max_tool_error_cascade=100, min_efficiency_score=0.0)
        long_text = "the quick brown fox jumps over the lazy dog and does many things"
        session = _make_session(num_blocks=7, block_text=long_text)
        # 6 pairs -> severity 3 = threshold
        stop, _ = should_stop(session, _baselines(), config)
        assert stop is True

    def test_cascade_exactly_at_threshold(self):
        """Exactly at threshold (>=) should stop."""
        config = EarlyStopConfig(max_looping_severity=100, max_tool_error_cascade=3, min_efficiency_score=0.0)
        calls = [_tc(i, is_error=True) for i in range(3)]
        session = _make_session(num_blocks=1, block_text="x", tool_calls=calls)
        stop, _ = should_stop(session, _baselines(), config)
        assert stop is True

    def test_efficiency_exactly_at_threshold(self):
        """Exactly at threshold (not <) should NOT stop."""
        # efficiency_score = 0.2 is not < 0.2, so should not stop
        # 1/(1+0.3*excess) = 0.2 -> excess = 13.33... not integer
        # With excess=13 (turns=23): 1/(1+3.9) = 1/4.9 ≈ 0.204 > 0.2 -> no stop
        config = EarlyStopConfig(max_looping_severity=100, max_tool_error_cascade=100, min_efficiency_score=0.2)
        session = _make_session(num_blocks=23, block_text="x")
        stop, _ = should_stop(session, _baselines(), config)
        # 0.204 > 0.2, so should NOT stop
        assert stop is False

    def test_all_below_threshold_no_stop(self):
        """No condition met -> no stop."""
        session = _make_session(num_blocks=5, block_text="x")
        stop, reason = should_stop(session, _baselines(), _config())
        assert stop is False
        assert reason == ""
