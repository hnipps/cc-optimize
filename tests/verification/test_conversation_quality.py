"""Verify compute_conversation_quality formula from spec Section 3.4 / after Section 4.9.

Four components weighted 0.25 each:
- Repetition: 1.0 at baseline, decreases 0.33 per severity level above
- Tool error: 1.0 at baseline, then 1/(1 + 0.5*excess)
- Repair: 1.0 at baseline, linear to 0.0 at frequency=1.0
- Efficiency: passes through efficiency_score directly
"""
from __future__ import annotations

import pytest

from cc_optimize.evaluation.evaluator import compute_conversation_quality
from cc_optimize.models.benchmark import SignalBaselines
from cc_optimize.models.signals import BehavioralSignals


def _baselines() -> SignalBaselines:
    return SignalBaselines(
        efficiency_turn_baseline=10,
        max_looping_severity=1,
        max_tool_error_cascade=2,
        max_repair_frequency=0.2,
    )


def _signals(**overrides) -> BehavioralSignals:
    """Create BehavioralSignals with defaults at/below baseline."""
    defaults = dict(
        efficiency_score=1.0,
        turn_count=10,
        repetition_count=0,
        repetition_exact_count=0,
        repetition_max_severity=0,  # below baseline of 1
        tool_error_max_cascade=1,   # below baseline of 2
        tool_error_total_failures=0,
        tool_error_total_calls=5,
        repair_frequency=0.1,       # below baseline of 0.2
        repair_count=0,
    )
    defaults.update(overrides)
    return BehavioralSignals(**defaults)


class TestAllAtBaseline:
    def test_all_at_or_below_baseline_gives_1(self):
        """When all signals are at or below baseline, quality = 1.0"""
        # Each component = 1.0, so 0.25*4 = 1.0
        baselines = _baselines()
        signals = _signals()
        quality = compute_conversation_quality(signals, baselines)
        assert quality == pytest.approx(1.0)


class TestRepetitionComponent:
    def test_severity_one_above_baseline(self):
        # baseline max_looping_severity=1, actual severity=2 -> excess=1
        # repetition_score = max(0, 1.0 - 0.33*1) = 0.67
        # other components = 1.0
        # quality = 0.25*0.67 + 0.25*1.0 + 0.25*1.0 + 0.25*1.0 = 0.9175
        baselines = _baselines()
        signals = _signals(repetition_max_severity=2)
        quality = compute_conversation_quality(signals, baselines)
        expected = 0.25 * 0.67 + 0.25 * 1.0 + 0.25 * 1.0 + 0.25 * 1.0
        assert quality == pytest.approx(expected, abs=0.01)

    def test_severity_three_above_baseline(self):
        # baseline=1, actual=4 (hypothetical), excess=3
        # repetition_score = max(0, 1.0 - 0.33*3) = max(0, 0.01) = 0.01
        baselines = _baselines()
        signals = _signals(repetition_max_severity=4)
        quality = compute_conversation_quality(signals, baselines)
        rep_score = max(0.0, 1.0 - 0.33 * 3)
        expected = 0.25 * rep_score + 0.25 * 1.0 + 0.25 * 1.0 + 0.25 * 1.0
        assert quality == pytest.approx(expected, abs=0.01)


class TestToolErrorComponent:
    def test_cascade_one_above_baseline(self):
        # baseline max_tool_error_cascade=2, actual=3 -> excess=1
        # tool_error_score = 1/(1+0.5*1) = 1/1.5 ≈ 0.667
        baselines = _baselines()
        signals = _signals(tool_error_max_cascade=3)
        quality = compute_conversation_quality(signals, baselines)
        tool_score = 1.0 / (1.0 + 0.5 * 1)
        expected = 0.25 * 1.0 + 0.25 * tool_score + 0.25 * 1.0 + 0.25 * 1.0
        assert quality == pytest.approx(expected, abs=0.001)

    def test_cascade_large_excess(self):
        # baseline=2, actual=12 -> excess=10
        # tool_error_score = 1/(1+5.0) = 1/6 ≈ 0.167
        baselines = _baselines()
        signals = _signals(tool_error_max_cascade=12)
        quality = compute_conversation_quality(signals, baselines)
        tool_score = 1.0 / (1.0 + 0.5 * 10)
        expected = 0.25 * 1.0 + 0.25 * tool_score + 0.25 * 1.0 + 0.25 * 1.0
        assert quality == pytest.approx(expected, abs=0.001)


class TestRepairComponent:
    def test_repair_above_baseline(self):
        # baseline max_repair_frequency=0.2, actual=0.5
        # remaining_range = 1.0 - 0.2 = 0.8
        # overshoot = 0.5 - 0.2 = 0.3
        # repair_score = max(0, 1.0 - 0.3/0.8) = max(0, 0.625) = 0.625
        baselines = _baselines()
        signals = _signals(repair_frequency=0.5)
        quality = compute_conversation_quality(signals, baselines)
        repair_score = 1.0 - 0.3 / 0.8
        expected = 0.25 * 1.0 + 0.25 * 1.0 + 0.25 * repair_score + 0.25 * 1.0
        assert quality == pytest.approx(expected, abs=0.001)

    def test_repair_at_frequency_1(self):
        # baseline=0.2, actual=1.0
        # overshoot = 0.8, remaining_range = 0.8
        # repair_score = max(0, 1.0 - 1.0) = 0.0
        baselines = _baselines()
        signals = _signals(repair_frequency=1.0)
        quality = compute_conversation_quality(signals, baselines)
        expected = 0.25 * 1.0 + 0.25 * 1.0 + 0.25 * 0.0 + 0.25 * 1.0
        assert quality == pytest.approx(expected, abs=0.001)


class TestEfficiencyComponent:
    def test_efficiency_passes_through(self):
        # efficiency_score=0.5, other components at baseline
        baselines = _baselines()
        signals = _signals(efficiency_score=0.5)
        quality = compute_conversation_quality(signals, baselines)
        expected = 0.25 * 1.0 + 0.25 * 1.0 + 0.25 * 1.0 + 0.25 * 0.5
        assert quality == pytest.approx(expected, abs=0.001)


class TestWorstCase:
    def test_all_worst(self):
        """All components at worst case."""
        baselines = _baselines()
        # severity=4 -> excess=3, rep_score=max(0, 1-0.99)=0.01
        # cascade=100 -> excess=98, tool_score=1/(1+49)=0.02
        # repair_freq=1.0 -> repair_score=0.0
        # efficiency=0.0
        signals = _signals(
            repetition_max_severity=4,
            tool_error_max_cascade=100,
            repair_frequency=1.0,
            efficiency_score=0.0,
        )
        quality = compute_conversation_quality(signals, baselines)
        rep = max(0.0, 1.0 - 0.33 * 3)
        tool = 1.0 / (1.0 + 0.5 * 98)
        repair = 0.0
        eff = 0.0
        expected = 0.25 * rep + 0.25 * tool + 0.25 * repair + 0.25 * eff
        assert quality == pytest.approx(expected, abs=0.001)
        assert quality < 0.1  # Sanity check: should be very low
