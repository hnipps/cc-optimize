"""End-to-end test exercising the full cc-optimize pipeline with real LLM calls."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from cc_optimize.cli import main


def _detect_branch(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.e2e
@pytest.mark.timeout(600)
class TestFullPipeline:
    """Runs seed → baseline → optimize → apply → report in sequence."""

    @pytest.fixture(autouse=True)
    def setup_repo_and_suite(self, tmp_path: Path):
        if not shutil.which("claude"):
            pytest.skip("claude CLI not found on PATH")

        # --- Set up a minimal git repo ---
        repo = tmp_path / "repo"
        repo.mkdir()

        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@test.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@test.com",
        }

        subprocess.run(["git", "init", str(repo)], check=True, capture_output=True, env=env)
        (repo / "README.md").write_text("# Test repo\n")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True, env=env)
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "init"],
            check=True, capture_output=True, env=env,
        )

        branch = _detect_branch(repo)

        # --- Write task YAML ---
        suite_dir = tmp_path / "suite"
        tasks_dir = suite_dir / "tasks"
        tasks_dir.mkdir(parents=True)

        task_yaml = {
            "id": "hello-greet",
            "name": "Create hello.py with greet function",
            "category": "feature",
            "prompt": (
                "Create a file called hello.py with a function greet(name) "
                "that returns f'Hello, {name}!'"
            ),
            "git_ref": branch,
            "timeout_seconds": 120,
            "signal_baselines": {
                "efficiency_turn_baseline": 5,
                "max_looping_severity": 0,
                "max_tool_error_cascade": 2,
                "max_repair_frequency": 0.15,
            },
            "success_criteria": [
                {
                    "type": "file_exists",
                    "path": "hello.py",
                    "description": "hello.py exists",
                },
                {
                    "type": "file_contains",
                    "path": "hello.py",
                    "pattern": r"def greet\(name\)",
                    "description": "greet function defined",
                },
                {
                    "type": "file_contains",
                    "path": "hello.py",
                    "pattern": r"Hello, \{name\}",
                    "description": "returns greeting string",
                },
            ],
            "tags": ["python", "trivial"],
        }
        with open(tasks_dir / "hello-task.yaml", "w") as f:
            yaml.dump(task_yaml, f, default_flow_style=False)

        suite_yaml = {
            "name": "e2e-mini",
            "repo_path": str(repo),
            "description": "Minimal E2E test suite",
            "tasks": ["tasks/hello-task.yaml"],
        }
        with open(suite_dir / "suite.yaml", "w") as f:
            yaml.dump(suite_yaml, f, default_flow_style=False)

        # Expose paths to the test method
        self.repo = repo
        self.suite_dir = suite_dir
        self.suite_path = suite_dir / "suite.yaml"
        self.tmp_path = tmp_path
        self.env = env

    def test_full_pipeline(self):
        runner = CliRunner()

        # --- Step 1: seed ---
        seed_output = self.tmp_path / "seed.yaml"
        result = runner.invoke(main, ["seed", str(self.repo), "-o", str(seed_output)])
        assert result.exit_code == 0, f"seed failed: {result.output}"
        assert seed_output.exists(), "seed.yaml not created"
        with open(seed_output) as f:
            seed_data = yaml.safe_load(f)
        assert "id" in seed_data
        assert "claude_md" in seed_data

        # --- Step 2: baseline ---
        baseline_dir = self.tmp_path / "baseline"
        result = runner.invoke(main, ["baseline", str(self.suite_path), "-o", str(baseline_dir)])
        assert result.exit_code == 0, f"baseline failed: {result.output}"
        traces_dir = baseline_dir / "traces"
        assert traces_dir.exists(), "traces/ dir not created"
        trace_files = list(traces_dir.glob("*.jsonl"))
        assert len(trace_files) >= 1, "No trace JSONL files created"

        # --- Step 3: optimize ---
        opt_dir = self.tmp_path / "optimization"
        result = runner.invoke(main, ["optimize", str(self.suite_path), "-n", "3", "-o", str(opt_dir)])
        assert result.exit_code == 0, f"optimize failed: {result.output}"
        assert (opt_dir / "report.md").exists(), "report.md not created"
        assert (opt_dir / "report.json").exists(), "report.json not created"
        assert (opt_dir / "best_candidate.yaml").exists(), "best_candidate.yaml not created"

        # Validate report.json content
        report_data = json.loads((opt_dir / "report.json").read_text())
        assert report_data["total_metric_calls"] <= 3, (
            f"Metric calls {report_data['total_metric_calls']} exceeded limit of 3"
        )
        assert report_data["total_metric_calls"] >= 1, "No metric calls recorded"
        assert len(report_data["optimization_trace"]) >= 1, "Empty optimization trace"

        # Validate best_candidate.yaml structure
        best_data = yaml.safe_load((opt_dir / "best_candidate.yaml").read_text())
        assert "id" in best_data, "best_candidate missing 'id'"
        assert "claude_md" in best_data, "best_candidate missing 'claude_md'"
        assert "rules" in best_data, "best_candidate missing 'rules'"
        assert "skills" in best_data, "best_candidate missing 'skills'"

        # --- Step 4: apply ---
        best_candidate = opt_dir / "best_candidate.yaml"
        result = runner.invoke(main, ["apply", str(best_candidate), str(self.repo)], input="y\n")
        assert result.exit_code == 0, f"apply failed: {result.output}"
        assert (self.repo / "CLAUDE.md").exists(), "CLAUDE.md not created in repo"
        claude_md_content = (self.repo / "CLAUDE.md").read_text()
        assert isinstance(claude_md_content, str), "CLAUDE.md content is not a string"

        # --- Step 5: report ---
        result = runner.invoke(main, ["report", str(opt_dir)])
        assert result.exit_code == 0, f"report failed: {result.output}"
        assert "# Optimization Report" in result.output, (
            f"Report output missing '# Optimization Report' heading. Got: {result.output[:200]}"
        )
