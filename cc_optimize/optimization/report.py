from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from cc_optimize.models.candidate import ConfigCandidate


@dataclass
class OptimizationReport:
    seed_candidate: ConfigCandidate
    best_candidate: ConfigCandidate
    pareto_front: list[ConfigCandidate]
    total_metric_calls: int
    total_tokens_consumed: int
    tokens_saved_by_early_stopping: int
    per_task_comparison: dict[str, dict]
    signal_improvements: dict[str, dict]
    optimization_trace: list[dict]
    wall_time_seconds: float

    def to_markdown(self) -> str:
        lines = ["# Optimization Report", ""]

        seed_components = self.seed_candidate.to_component_dict()
        best_components = self.best_candidate.to_component_dict()
        seed_score = self._trace_first_score()
        best_score = self._trace_best_score()

        lines.append("## Summary")
        lines.append(f"- Seed score: {seed_score:.3f}")
        lines.append(f"- Best score: {best_score:.3f}")
        lines.append(f"- Delta: {best_score - seed_score:+.3f}")
        lines.append(f"- Total metric calls: {self.total_metric_calls}")
        lines.append(f"- Wall time: {self.wall_time_seconds:.1f}s")
        lines.append(f"- Total tokens consumed: {self.total_tokens_consumed}")
        lines.append("")

        if self.per_task_comparison:
            lines.append("## Per-Task Comparison")
            lines.append("| Task | Seed Score | Best Score | Delta |")
            lines.append("|------|-----------|-----------|-------|")
            for task_id, comp in self.per_task_comparison.items():
                ss = comp.get("seed_score", 0)
                bs = comp.get("best_score", 0)
                lines.append(f"| {task_id} | {ss:.3f} | {bs:.3f} | {bs - ss:+.3f} |")
            lines.append("")

        if self.signal_improvements:
            lines.append("## Signal Improvements")
            for signal_name, imp in self.signal_improvements.items():
                before = imp.get("before", 0)
                after = imp.get("after", 0)
                lines.append(f"- {signal_name}: {before:.3f} → {after:.3f} ({after - before:+.3f})")
            lines.append("")

        if self.optimization_trace:
            lines.append("## Top Mutations")
            sorted_trace = sorted(
                self.optimization_trace, key=lambda x: x.get("score", 0), reverse=True
            )
            for entry in sorted_trace[:3]:
                lines.append(
                    f"- Iteration {entry.get('iteration', '?')}: "
                    f"score={entry.get('score', 0):.3f}"
                )
            lines.append("")

        lines.append("## Configuration Diff")
        all_keys = set(seed_components) | set(best_components)
        for key in sorted(all_keys):
            seed_val = seed_components.get(key, "<not present>")
            best_val = best_components.get(key, "<not present>")
            if seed_val != best_val:
                lines.append(f"### {key}")
                lines.append(f"**Before:**\n```\n{seed_val}\n```")
                lines.append(f"**After:**\n```\n{best_val}\n```")
                lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "seed_id": self.seed_candidate.id,
                "best_id": self.best_candidate.id,
                "total_metric_calls": self.total_metric_calls,
                "total_tokens_consumed": self.total_tokens_consumed,
                "tokens_saved_by_early_stopping": self.tokens_saved_by_early_stopping,
                "per_task_comparison": self.per_task_comparison,
                "signal_improvements": self.signal_improvements,
                "optimization_trace": self.optimization_trace,
                "wall_time_seconds": self.wall_time_seconds,
            },
            indent=2,
        )

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.md").write_text(self.to_markdown())
        (output_dir / "report.json").write_text(self.to_json())

        best_yaml = {
            "id": self.best_candidate.id,
            "claude_md": self.best_candidate.claude_md,
            "rules": [
                {
                    "filename": r.filename,
                    "content": r.content,
                    "paths": r.paths,
                }
                for r in self.best_candidate.rules
            ],
            "skills": [
                {"name": s.name, "skill_md": s.skill_md}
                for s in self.best_candidate.skills
            ],
            "settings_json": self.best_candidate.settings_json,
            "context_imports": self.best_candidate.context_imports,
        }
        import yaml

        (output_dir / "best_candidate.yaml").write_text(
            yaml.dump(best_yaml, default_flow_style=False)
        )

    def _trace_first_score(self) -> float:
        if self.optimization_trace:
            return self.optimization_trace[0].get("score", 0.0)
        return 0.0

    def _trace_best_score(self) -> float:
        if self.optimization_trace:
            return max(e.get("score", 0.0) for e in self.optimization_trace)
        return 0.0
