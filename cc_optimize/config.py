from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ClaudeCliConfig:
    command: str = "claude"
    extra_flags: list[str] = field(default_factory=list)


@dataclass
class EarlyStopConfig:
    max_looping_severity: int = 3
    max_tool_error_cascade: int = 5
    min_efficiency_score: float = 0.2
    check_interval_events: int = 10


@dataclass
class OptimizationConfig:
    max_metric_calls: int = 150
    minibatch_size: int = 4
    stop_after_no_improvement: int = 20
    score_threshold: float = 0.95
    score_weights: tuple[float, float, float] = (0.5, 0.25, 0.25)
    reflection_model: str = "anthropic/claude-sonnet-4-20250514"
    num_trials: int = 1


@dataclass
class GlobalConfig:
    work_dir: Path = field(default_factory=lambda: Path("/tmp/cc-optimize"))
    claude_cli: ClaudeCliConfig = field(default_factory=ClaudeCliConfig)
    early_stopping: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> GlobalConfig:
        search_paths = []
        if path:
            search_paths.append(path)
        search_paths.extend([
            Path("./cc-optimize.yaml"),
            Path.home() / ".cc-optimize" / "config.yaml",
        ])
        for p in search_paths:
            if p.exists():
                with open(p) as f:
                    data = yaml.safe_load(f) or {}
                return cls._from_dict(data)
        return cls()

    @classmethod
    def _from_dict(cls, data: dict) -> GlobalConfig:
        config = cls()
        if "work_dir" in data:
            config.work_dir = Path(data["work_dir"])
        if "claude_cli" in data:
            cli = data["claude_cli"]
            config.claude_cli = ClaudeCliConfig(
                command=cli.get("command", "claude"),
                extra_flags=cli.get("extra_flags", []),
            )
        if "early_stopping" in data:
            es = data["early_stopping"]
            config.early_stopping = EarlyStopConfig(
                max_looping_severity=es.get("max_looping_severity", 3),
                max_tool_error_cascade=es.get("max_tool_error_cascade", 5),
                min_efficiency_score=es.get("min_efficiency_score", 0.2),
                check_interval_events=es.get("check_interval_events", 10),
            )
        if "optimization" in data:
            opt = data["optimization"]
            weights = opt.get("score_weights", [0.5, 0.25, 0.25])
            config.optimization = OptimizationConfig(
                max_metric_calls=opt.get("max_metric_calls", 150),
                minibatch_size=opt.get("minibatch_size", 4),
                stop_after_no_improvement=opt.get("stop_after_no_improvement", 20),
                score_threshold=opt.get("score_threshold", 0.95),
                score_weights=tuple(weights),
                reflection_model=opt.get(
                    "reflection_model", "anthropic/claude-sonnet-4-20250514"
                ),
                num_trials=opt.get("num_trials", 1),
            )
        return config

    def save(self, path: Path) -> None:
        data = {
            "work_dir": str(self.work_dir),
            "claude_cli": {
                "command": self.claude_cli.command,
                "extra_flags": self.claude_cli.extra_flags,
            },
            "early_stopping": {
                "max_looping_severity": self.early_stopping.max_looping_severity,
                "max_tool_error_cascade": self.early_stopping.max_tool_error_cascade,
                "min_efficiency_score": self.early_stopping.min_efficiency_score,
                "check_interval_events": self.early_stopping.check_interval_events,
            },
            "optimization": {
                "max_metric_calls": self.optimization.max_metric_calls,
                "minibatch_size": self.optimization.minibatch_size,
                "stop_after_no_improvement": self.optimization.stop_after_no_improvement,
                "score_threshold": self.optimization.score_threshold,
                "score_weights": list(self.optimization.score_weights),
                "reflection_model": self.optimization.reflection_model,
                "num_trials": self.optimization.num_trials,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
