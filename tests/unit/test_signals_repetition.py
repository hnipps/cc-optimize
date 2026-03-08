from __future__ import annotations

from cc_optimize.signals.jsonl_parser import AssistantBlock, ParsedSession, ToolCall
from cc_optimize.signals.repetition import bigram_jaccard, compute_repetition


def make_session(assistant_blocks=None, tool_calls=None):
    return ParsedSession(
        assistant_blocks=assistant_blocks or [],
        all_tool_calls=tool_calls or [],
        total_input_tokens=0,
        total_output_tokens=0,
        raw_events=[],
    )


LONG_TEXT = "the quick brown fox jumps over the lazy dog near the river bank"
DIFFERENT_TEXT = "python programming language offers many powerful features for developers today"


class TestBigramJaccard:
    def test_identical_texts(self):
        assert bigram_jaccard(LONG_TEXT, LONG_TEXT) == 1.0

    def test_completely_different(self):
        sim = bigram_jaccard(LONG_TEXT, DIFFERENT_TEXT)
        assert sim < 0.1

    def test_both_empty(self):
        assert bigram_jaccard("", "") == 0.0

    def test_one_empty(self):
        assert bigram_jaccard(LONG_TEXT, "") == 0.0

    def test_single_word_each(self):
        # Single words produce no bigrams
        assert bigram_jaccard("hello", "hello") == 0.0


class TestComputeRepetition:
    def test_two_identical_long_blocks(self):
        blocks = [
            AssistantBlock(text=LONG_TEXT, tool_calls=[], index=0),
            AssistantBlock(text=LONG_TEXT, tool_calls=[], index=1),
        ]
        session = make_session(assistant_blocks=blocks)
        count, exact, severity = compute_repetition(session)
        assert exact == 1
        assert count == 1
        assert severity == 1

    def test_two_unrelated_blocks(self):
        blocks = [
            AssistantBlock(text=LONG_TEXT, tool_calls=[], index=0),
            AssistantBlock(text=DIFFERENT_TEXT, tool_calls=[], index=1),
        ]
        session = make_session(assistant_blocks=blocks)
        count, exact, severity = compute_repetition(session)
        assert count == 0
        assert exact == 0
        assert severity == 0

    def test_seven_identical_blocks_severity_3(self):
        blocks = [
            AssistantBlock(text=LONG_TEXT, tool_calls=[], index=i) for i in range(7)
        ]
        session = make_session(assistant_blocks=blocks)
        count, exact, severity = compute_repetition(session)
        # 6 consecutive pairs, all identical
        assert count == 6
        assert exact == 6
        assert severity == 3

    def test_all_blocks_under_five_words(self):
        blocks = [
            AssistantBlock(text="too short", tool_calls=[], index=0),
            AssistantBlock(text="also short", tool_calls=[], index=1),
        ]
        session = make_session(assistant_blocks=blocks)
        count, exact, severity = compute_repetition(session)
        assert count == 0
        assert exact == 0
        assert severity == 0

    def test_single_block_session(self):
        blocks = [AssistantBlock(text=LONG_TEXT, tool_calls=[], index=0)]
        session = make_session(assistant_blocks=blocks)
        count, exact, severity = compute_repetition(session)
        assert count == 0
        assert exact == 0
        assert severity == 0

    def test_empty_session(self):
        session = make_session(assistant_blocks=[])
        count, exact, severity = compute_repetition(session)
        assert count == 0
        assert exact == 0
        assert severity == 0
