"""Data flow smoke test: verify types connect through the full pipeline.

Without any mocking, builds data programmatically and passes it through
every layer: JSONL parse -> signals -> metrics -> TaskResult -> MinibatchResult.
"""
from __future__ import annotations

import json
from pathlib import Path

from cc_optimize.evaluation.evaluator import evaluate_minibatch
from cc_optimize.evaluation.metrics import compute_coding_metrics
from cc_optimize.models.benchmark import (
    BenchmarkTask,
    SignalBaselines,
    SuccessCriterion,
    TaskCategory,
)
from cc_optimize.models.candidate import ConfigCandidate
from cc_optimize.models.result import TaskResult
from cc_optimize.signals.compute import compute_all_signals
from cc_optimize.signals.jsonl_parser import parse


class TestFullDataFlow:
    def test_types_connect_end_to_end(self, tmp_path):
        """Build -> parse -> compute -> evaluate -> score, all types connected."""

        # 1. Build a BenchmarkTask programmatically
        baselines = SignalBaselines(
            efficiency_turn_baseline=5,
            max_looping_severity=0,
            max_tool_error_cascade=2,
            max_repair_frequency=0.2,
        )
        task = BenchmarkTask(
            id="smoke-test",
            name="Smoke Test",
            category=TaskCategory.BUG_FIX,
            prompt="fix the bug",
            repo_path=tmp_path,
            git_ref="HEAD",
            success_criteria=[
                SuccessCriterion(type="file_exists", path="output.txt", description="output created"),
            ],
            signal_baselines=baselines,
        )

        # 2. Build a ConfigCandidate programmatically
        candidate = ConfigCandidate(
            id="cand-smoke",
            claude_md="# Test Config\nBe efficient.",
        )

        # 3. Write a minimal JSONL file
        jsonl_path = tmp_path / "trace.jsonl"
        events = [
            {"type": "system", "subtype": "init", "session_id": "s1"},
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "I'll fix the bug now"},
                {"type": "tool_use", "id": "tc_1", "name": "Read", "input": {"path": "src/main.py"}},
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc_1", "content": "def main(): pass", "is_error": False},
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Found the issue, fixing now"},
                {"type": "tool_use", "id": "tc_2", "name": "Edit", "input": {"file_path": "src/main.py"}},
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc_2", "content": "edited", "is_error": False},
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Running tests"},
                {"type": "tool_use", "id": "tc_3", "name": "Bash", "input": {"command": "pytest"}},
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tc_3", "content": "PASS", "is_error": False},
            ]}},
            {"type": "result", "result": "Bug fixed", "usage": {"input_tokens": 2000, "output_tokens": 800}},
        ]
        jsonl_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

        # 4. Call the JSONL parser
        session = parse(jsonl_path)
        assert len(session.assistant_blocks) == 3
        assert len(session.all_tool_calls) == 3

        # 5. Call compute_all_signals
        signals = compute_all_signals(session, baselines)
        assert 0.0 <= signals.efficiency_score <= 1.0
        assert signals.turn_count == 3
        assert signals.tool_error_total_calls == 3
        assert signals.tool_error_total_failures == 0

        # 6. Call compute_coding_metrics
        metrics = compute_coding_metrics(session)
        assert metrics["tokens_input"] == 2000
        assert metrics["tokens_output"] == 800
        assert metrics["tokens_total"] == 2800
        assert metrics["file_edit_churn"] >= 0.0
        assert metrics["tool_error_rate"] == 0.0

        # 7. Build a TaskResult from these outputs
        task_result = TaskResult(
            task_id=task.id,
            candidate_id=candidate.id,
            success=True,
            criteria_results={"output created": True},
            signals=signals,
            tokens_input=metrics["tokens_input"],
            tokens_output=metrics["tokens_output"],
            tokens_total=metrics["tokens_total"],
            wall_time_seconds=10.0,
            file_edit_churn=metrics["file_edit_churn"],
            tool_error_rate=metrics["tool_error_rate"],
            session_trace_path=jsonl_path,
        )

        # 8. Call evaluate_minibatch
        minibatch_result = evaluate_minibatch(
            tasks=[task],
            candidate=candidate,
            task_results=[task_result],
        )

        # 9. Assert MinibatchResult fields
        assert minibatch_result.candidate_id == candidate.id
        assert len(minibatch_result.task_results) == 1

        # CompositeScore values in [0, 1]
        cs = minibatch_result.composite_score
        assert 0.0 <= cs.correctness <= 1.0
        assert 0.0 <= cs.efficiency <= 1.0
        assert 0.0 <= cs.conversation_quality <= 1.0

        # Weighted scalar in [0, 1]
        ws = cs.weighted_scalar()
        assert 0.0 <= ws <= 1.0

        # Actionable side info is non-empty
        assert isinstance(minibatch_result.actionable_side_info, str)
        assert len(minibatch_result.actionable_side_info) > 0
