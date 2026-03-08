from __future__ import annotations

from pathlib import Path

from cc_optimize.models.candidate import ConfigCandidate, RuleFile
from cc_optimize.optimization.report import OptimizationReport


def _make_report() -> OptimizationReport:
    seed = ConfigCandidate(id="seed", claude_md="Be helpful.")
    best = ConfigCandidate(
        id="best",
        claude_md="Be helpful and concise.",
        rules=[RuleFile(filename="go.md", content="Use gofmt.", paths=["*.go"])],
    )
    return OptimizationReport(
        seed_candidate=seed,
        best_candidate=best,
        pareto_front=[best],
        total_metric_calls=50,
        total_tokens_consumed=100000,
        tokens_saved_by_early_stopping=5000,
        per_task_comparison={
            "task-1": {"seed_score": 0.5, "best_score": 0.8},
        },
        signal_improvements={
            "efficiency": {"before": 0.6, "after": 0.9},
        },
        optimization_trace=[
            {"iteration": 0, "score": 0.5},
            {"iteration": 1, "score": 0.7},
            {"iteration": 2, "score": 0.8},
        ],
        wall_time_seconds=120.5,
    )


class TestOptimizationReport:
    def test_to_markdown(self):
        report = _make_report()
        md = report.to_markdown()
        assert "# Optimization Report" in md
        assert "Seed score: 0.500" in md
        assert "Best score: 0.800" in md
        assert "| task-1 | 0.500 | 0.800 | +0.300 |" in md
        assert "efficiency: 0.600" in md
        assert "0.900" in md

    def test_to_json(self):
        import json

        report = _make_report()
        data = json.loads(report.to_json())
        assert data["total_metric_calls"] == 50
        assert "task-1" in data["per_task_comparison"]

    def test_save(self, tmp_path: Path):
        import json
        import yaml

        report = _make_report()
        report.save(tmp_path / "output")

        md_content = (tmp_path / "output" / "report.md").read_text()
        assert "# Optimization Report" in md_content

        json_data = json.loads((tmp_path / "output" / "report.json").read_text())
        assert json_data["total_metric_calls"] == 50
        assert "task-1" in json_data["per_task_comparison"]

        yaml_data = yaml.safe_load((tmp_path / "output" / "best_candidate.yaml").read_text())
        assert yaml_data["claude_md"] == "Be helpful and concise."
        assert yaml_data["rules"][0]["filename"] == "go.md"
