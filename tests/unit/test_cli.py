from __future__ import annotations

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from cc_optimize.cli import main


class TestSeedCommand:
    def test_seed_reads_config(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "CLAUDE.md").write_text("# My Config")

        runner = CliRunner()
        output = tmp_path / "seed.yaml"
        result = runner.invoke(main, ["seed", str(repo), "--output", str(output)])
        assert result.exit_code == 0
        assert "Read config from" in result.output
        assert output.exists()

        with open(output) as f:
            data = yaml.safe_load(f)
        assert data["claude_md"] == "# My Config"


class TestApplyCommand:
    def test_apply_writes_config(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()

        candidate_data = {
            "id": "test-apply",
            "claude_md": "Applied config.",
            "rules": [],
            "skills": [],
            "settings_json": {},
            "context_imports": [],
        }
        candidate_path = tmp_path / "candidate.yaml"
        with open(candidate_path, "w") as f:
            yaml.dump(candidate_data, f)

        runner = CliRunner()
        result = runner.invoke(
            main, ["apply", str(candidate_path), str(repo)], input="y\n"
        )
        assert result.exit_code == 0
        assert (repo / "CLAUDE.md").read_text() == "Applied config."


class TestReportCommand:
    def test_report_displays(self, tmp_path: Path):
        (tmp_path / "report.md").write_text("# Test Report\nContent here.")
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(tmp_path)])
        assert result.exit_code == 0
        assert "Test Report" in result.output

    def test_report_missing(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(tmp_path)])
        assert result.exit_code != 0
