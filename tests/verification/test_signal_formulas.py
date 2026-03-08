"""Verify every signal formula from spec Section 4.2 with exact test cases."""
from __future__ import annotations

import pytest

from cc_optimize.signals.efficiency import compute_efficiency_score
from cc_optimize.signals.jsonl_parser import AssistantBlock, ParsedSession, ToolCall
from cc_optimize.signals.repair import compute_repair_frequency
from cc_optimize.signals.repetition import bigram_jaccard, compute_repetition
from cc_optimize.signals.tool_errors import compute_tool_error_cascade


# ---------------------------------------------------------------------------
# Helpers to build ParsedSession objects without JSONL files
# ---------------------------------------------------------------------------

def _make_session(
    assistant_texts: list[str] | None = None,
    tool_calls: list[ToolCall] | None = None,
) -> ParsedSession:
    """Build a ParsedSession programmatically."""
    blocks: list[AssistantBlock] = []
    if assistant_texts:
        for i, text in enumerate(assistant_texts):
            blocks.append(AssistantBlock(text=text, tool_calls=[], index=i))
    return ParsedSession(
        assistant_blocks=blocks,
        all_tool_calls=tool_calls or [],
        total_input_tokens=0,
        total_output_tokens=0,
        raw_events=[],
    )


def _make_tool_call(
    index: int, tool_name: str = "Bash", is_error: bool = False
) -> ToolCall:
    return ToolCall(
        tool_use_id=f"tc_{index}",
        tool_name=tool_name,
        input_data={},
        output=None,
        is_error=is_error,
        timestamp_index=index,
    )


# ===========================================================================
# 1. Efficiency (Section 4.2.1)
# ===========================================================================

class TestEfficiency:
    """Spec Section 4.2.1 — exact test table."""

    def test_at_baseline(self):
        # turn_count=15, baseline=15 -> 1.0
        session = _make_session(assistant_texts=["x"] * 15)
        score, turn_count = compute_efficiency_score(session, baseline_turns=15)
        assert turn_count == 15
        assert score == 1.0

    def test_over_baseline(self):
        # turn_count=25, baseline=15 -> 1/(1+3.0) = 0.25
        session = _make_session(assistant_texts=["x"] * 25)
        score, turn_count = compute_efficiency_score(session, baseline_turns=15)
        assert turn_count == 25
        assert score == pytest.approx(0.25)

    def test_under_baseline(self):
        # turn_count=10, baseline=15 -> 1.0 (cap, don't reward under-baseline)
        session = _make_session(assistant_texts=["x"] * 10)
        score, turn_count = compute_efficiency_score(session, baseline_turns=15)
        assert turn_count == 10
        assert score == 1.0

    def test_zero_turns(self):
        # turn_count=0, baseline=15 -> 1.0 (edge case)
        session = _make_session(assistant_texts=[])
        score, turn_count = compute_efficiency_score(session, baseline_turns=15)
        assert turn_count == 0
        assert score == 1.0

    def test_one_over_baseline(self):
        # turn_count=16, baseline=15 -> 1/(1+0.3) ≈ 0.769
        session = _make_session(assistant_texts=["x"] * 16)
        score, turn_count = compute_efficiency_score(session, baseline_turns=15)
        assert turn_count == 16
        assert score == pytest.approx(1.0 / 1.3, rel=1e-3)


# ===========================================================================
# 2. Repetition (Section 4.2.2) — bigram Jaccard + severity
# ===========================================================================

class TestBigramJaccard:
    def test_identical_texts(self):
        text = "the quick brown fox jumps over the lazy dog multiple times here"
        assert bigram_jaccard(text, text) == 1.0

    def test_completely_different(self):
        a = "the quick brown fox jumps over the lazy dog today"
        b = "alpha bravo charlie delta echo foxtrot golf hotel india"
        sim = bigram_jaccard(a, b)
        assert sim == pytest.approx(0.0, abs=0.05)

    def test_empty_texts(self):
        assert bigram_jaccard("", "") == 0.0

    def test_single_word(self):
        # Single words produce no bigrams
        assert bigram_jaccard("hello", "hello") == 0.0


class TestRepetition:
    def test_two_identical_long_blocks(self):
        text = "the quick brown fox jumps over the lazy dog and does many things"
        session = _make_session(assistant_texts=[text, text])
        count, exact, severity = compute_repetition(session)
        assert count == 1
        assert exact == 1
        assert severity == 1  # 1-2 -> severity 1

    def test_two_unrelated_blocks(self):
        a = "the quick brown fox jumps over the lazy dog today"
        b = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        session = _make_session(assistant_texts=[a, b])
        count, exact, severity = compute_repetition(session)
        assert count == 0
        assert exact == 0
        assert severity == 0

    def test_seven_near_identical_blocks(self):
        # 7 consecutive near-identical blocks -> 6 pairs -> severity 3
        text = "the quick brown fox jumps over the lazy dog and does many things"
        session = _make_session(assistant_texts=[text] * 7)
        count, exact, severity = compute_repetition(session)
        assert count == 6
        assert exact == 6
        assert severity == 3  # >=6 -> severity 3

    def test_all_short_blocks(self):
        # All blocks < 5 words -> skipped
        session = _make_session(assistant_texts=["one two", "three four", "five six"])
        count, exact, severity = compute_repetition(session)
        assert count == 0
        assert exact == 0
        assert severity == 0

    def test_single_block(self):
        text = "the quick brown fox jumps over the lazy dog and does many things"
        session = _make_session(assistant_texts=[text])
        count, exact, severity = compute_repetition(session)
        assert count == 0
        assert exact == 0
        assert severity == 0

    def test_severity_2_range(self):
        # 3-5 near-identical pairs -> severity 2
        text = "the quick brown fox jumps over the lazy dog and does many things"
        # 4 blocks = 3 consecutive pairs
        session = _make_session(assistant_texts=[text] * 4)
        count, exact, severity = compute_repetition(session)
        assert count == 3
        assert severity == 2


# ===========================================================================
# 3. Tool errors (Section 4.2.3)
# ===========================================================================

class TestToolErrorCascade:
    """Spec Section 4.2.3 — exact cascade sequences."""

    def test_all_success(self):
        # S, S, S -> max_cascade=0
        calls = [_make_tool_call(i) for i in range(3)]
        session = _make_session(tool_calls=calls)
        max_c, failures, total = compute_tool_error_cascade(session)
        assert max_c == 0
        assert failures == 0
        assert total == 3

    def test_single_failure_then_success(self):
        # F, S -> max_cascade=1
        calls = [
            _make_tool_call(0, is_error=True),
            _make_tool_call(1),
        ]
        session = _make_session(tool_calls=calls)
        max_c, failures, total = compute_tool_error_cascade(session)
        assert max_c == 1
        assert failures == 1
        assert total == 2

    def test_five_failures_then_success(self):
        # F, F, F, F, F, S -> max_cascade=5
        calls = [_make_tool_call(i, is_error=True) for i in range(5)]
        calls.append(_make_tool_call(5))
        session = _make_session(tool_calls=calls)
        max_c, failures, total = compute_tool_error_cascade(session)
        assert max_c == 5
        assert failures == 5
        assert total == 6

    def test_mixed_cascades(self):
        # S, F, F, S, F, S -> max_cascade=2
        calls = [
            _make_tool_call(0),
            _make_tool_call(1, is_error=True),
            _make_tool_call(2, is_error=True),
            _make_tool_call(3),
            _make_tool_call(4, is_error=True),
            _make_tool_call(5),
        ]
        session = _make_session(tool_calls=calls)
        max_c, failures, total = compute_tool_error_cascade(session)
        assert max_c == 2
        assert failures == 3
        assert total == 6

    def test_trailing_failures(self):
        # F, F, F (no trailing success) -> max_cascade=3
        calls = [_make_tool_call(i, is_error=True) for i in range(3)]
        session = _make_session(tool_calls=calls)
        max_c, failures, total = compute_tool_error_cascade(session)
        assert max_c == 3
        assert failures == 3
        assert total == 3

    def test_empty(self):
        # (empty) -> max_cascade=0
        session = _make_session(tool_calls=[])
        max_c, failures, total = compute_tool_error_cascade(session)
        assert max_c == 0
        assert failures == 0
        assert total == 0


# ===========================================================================
# 4. Repair frequency (Section 4.2.4)
# ===========================================================================

class TestRepairFrequency:
    """Spec Section 4.2.4 — exact repair sequences."""

    def test_classic_repair(self):
        # Bash(fail), Bash(success) -> repair_count=1
        calls = [
            _make_tool_call(0, tool_name="Bash", is_error=True),
            _make_tool_call(1, tool_name="Bash"),
        ]
        session = _make_session(tool_calls=calls)
        freq, count = compute_repair_frequency(session)
        assert count == 1
        assert freq == pytest.approx(0.5)

    def test_different_tool_not_repair(self):
        # Bash(fail), Write(success) -> repair_count=0 (different tool)
        calls = [
            _make_tool_call(0, tool_name="Bash", is_error=True),
            _make_tool_call(1, tool_name="Write"),
        ]
        session = _make_session(tool_calls=calls)
        freq, count = compute_repair_frequency(session)
        assert count == 0
        assert freq == pytest.approx(0.0)

    def test_double_failure_repair(self):
        # Bash(fail), Bash(fail), Bash(success) -> repair_count=2
        # Both fail->same-tool transitions count
        calls = [
            _make_tool_call(0, tool_name="Bash", is_error=True),
            _make_tool_call(1, tool_name="Bash", is_error=True),
            _make_tool_call(2, tool_name="Bash"),
        ]
        session = _make_session(tool_calls=calls)
        freq, count = compute_repair_frequency(session)
        assert count == 2
        assert freq == pytest.approx(2.0 / 3.0)

    def test_success_to_success_not_repair(self):
        # Bash(success), Bash(success) -> repair_count=0
        calls = [
            _make_tool_call(0, tool_name="Bash"),
            _make_tool_call(1, tool_name="Bash"),
        ]
        session = _make_session(tool_calls=calls)
        freq, count = compute_repair_frequency(session)
        assert count == 0
        assert freq == pytest.approx(0.0)

    def test_empty(self):
        session = _make_session(tool_calls=[])
        freq, count = compute_repair_frequency(session)
        assert count == 0
        assert freq == pytest.approx(0.0)
