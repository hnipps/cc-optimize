from __future__ import annotations

from pathlib import Path

from cc_optimize.config import GlobalConfig


class TestGlobalConfig:
    def test_defaults(self):
        config = GlobalConfig()
        assert config.work_dir == Path("/tmp/cc-optimize")
        assert config.claude_cli.command == "claude"
        assert config.early_stopping.max_looping_severity == 3
        assert config.optimization.max_metric_calls == 150

    def test_save_and_load(self, tmp_path: Path):
        config = GlobalConfig()
        config.work_dir = tmp_path / "work"
        config.optimization.max_metric_calls = 50
        config_path = tmp_path / "config.yaml"
        config.save(config_path)
        loaded = GlobalConfig.load(config_path)
        assert loaded.work_dir == config.work_dir
        assert loaded.optimization.max_metric_calls == 50

    def test_load_missing_returns_defaults(self):
        config = GlobalConfig.load(Path("/nonexistent/path.yaml"))
        assert config.optimization.max_metric_calls == 150
