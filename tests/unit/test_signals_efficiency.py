from __future__ import annotations

import pytest

from cc_optimize.signals.efficiency import compute_efficiency_score
from cc_optimize.signals.jsonl_parser import AssistantBlock, ParsedSession, ToolCall


def make_session(assistant_blocks=None, tool_calls=None):
    return ParsedSession(
        assistant_blocks=assistant_blocks or [],
        all_tool_calls=tool_calls or [],
        total_input_tokens=0,
        total_output_tokens=0,
        raw_events=[],
    )


def _blocks(n: int) -> list[AssistantBlock]:
    return [AssistantBlock(text=f"block {i}", tool_calls=[], index=i) for i in range(n)]


class TestComputeEfficiencyScore:
    def test_at_baseline(self):
        session = make_session(assistant_blocks=_blocks(15))
        score, count = compute_efficiency_score(session, baseline_turns=15)
        assert count == 15
        assert score == 1.0

    def test_above_baseline(self):
        session = make_session(assistant_blocks=_blocks(25))
        score, count = compute_efficiency_score(session, baseline_turns=15)
        assert count == 25
        assert score == pytest.approx(1.0 / (1.0 + 3.0))  # 0.25

    def test_below_baseline(self):
        session = make_session(assistant_blocks=_blocks(10))
        score, count = compute_efficiency_score(session, baseline_turns=15)
        assert count == 10
        assert score == 1.0

    def test_zero_turns(self):
        session = make_session(assistant_blocks=[])
        score, count = compute_efficiency_score(session, baseline_turns=15)
        assert count == 0
        assert score == 1.0

    def test_one_above_baseline(self):
        session = make_session(assistant_blocks=_blocks(16))
        score, count = compute_efficiency_score(session, baseline_turns=15)
        assert count == 16
        assert score == pytest.approx(1.0 / (1.0 + 0.3))  # ~0.769
