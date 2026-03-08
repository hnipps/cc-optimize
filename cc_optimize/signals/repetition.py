from __future__ import annotations

import re

from cc_optimize.signals.jsonl_parser import ParsedSession


def bigram_jaccard(text_a: str, text_b: str) -> float:
    """Jaccard similarity between word bigram sets.
    1. Tokenize: re.findall(r'\\b\\w+\\b', text.lower())
    2. Build bigram sets: {(words[i], words[i+1]) for i in range(len(words)-1)}
    3. Return |intersection| / |union| (0.0 if both empty)
    """
    words_a = re.findall(r"\b\w+\b", text_a.lower())
    words_b = re.findall(r"\b\w+\b", text_b.lower())
    bigrams_a = {(words_a[i], words_a[i + 1]) for i in range(len(words_a) - 1)}
    bigrams_b = {(words_b[i], words_b[i + 1]) for i in range(len(words_b) - 1)}
    if not bigrams_a and not bigrams_b:
        return 0.0
    intersection = bigrams_a & bigrams_b
    union = bigrams_a | bigrams_b
    return len(intersection) / len(union)


def compute_repetition(session: ParsedSession) -> tuple[int, int, int]:
    """Returns (repetition_count, exact_count, max_severity).
    For each consecutive pair of assistant text blocks (i, i+1):
    - Skip blocks with fewer than 5 words
    - Compute bigram_jaccard
    - If >= 0.85: exact_count += 1, repetition_count += 1
    - If 0.50 <= sim < 0.85: repetition_count += 1
    Severity: 0 if count==0, 1 if 1-2, 2 if 3-5, 3 if >=6
    """
    repetition_count = 0
    exact_count = 0

    blocks = session.assistant_blocks
    for i in range(len(blocks) - 1):
        text_a = blocks[i].text
        text_b = blocks[i + 1].text
        words_a = re.findall(r"\b\w+\b", text_a.lower())
        words_b = re.findall(r"\b\w+\b", text_b.lower())
        if len(words_a) < 5 or len(words_b) < 5:
            continue
        sim = bigram_jaccard(text_a, text_b)
        if sim >= 0.85:
            exact_count += 1
            repetition_count += 1
        elif sim >= 0.50:
            repetition_count += 1

    if repetition_count == 0:
        max_severity = 0
    elif repetition_count <= 2:
        max_severity = 1
    elif repetition_count <= 5:
        max_severity = 2
    else:
        max_severity = 3

    return repetition_count, exact_count, max_severity
