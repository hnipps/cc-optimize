from __future__ import annotations

from pathlib import Path

import click
import yaml


@click.group()
def main():
    """cc-optimize: GEPA-driven Claude Code configuration optimizer."""
    pass


@main.command()
@click.argument("repo_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", default="seed.yaml", type=click.Path(path_type=Path),
              help="Output path for seed candidate YAML.")
def seed(repo_path: Path, output: Path):
    """Read current Claude Code config from a repo and save as seed candidate."""
    from cc_optimize.adapter.config_reader import read_config

    candidate = read_config(repo_path)

    click.echo(f"Read config from {repo_path}")
    click.echo(f"  CLAUDE.md: {len(candidate.claude_md)} chars")
    click.echo(f"  Rules: {len(candidate.rules)}")
    click.echo(f"  Skills: {len(candidate.skills)}")
    click.echo(f"  Context imports: {len(candidate.context_imports)}")

    data = {
        "id": candidate.id,
        "claude_md": candidate.claude_md,
        "rules": [
            {"filename": r.filename, "content": r.content, "paths": r.paths}
            for r in candidate.rules
        ],
        "skills": [
            {"name": s.name, "skill_md": s.skill_md}
            for s in candidate.skills
        ],
        "settings_json": candidate.settings_json,
        "context_imports": candidate.context_imports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    click.echo(f"Saved seed candidate to {output}")


@main.command()
@click.argument("suite_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", "-o", default="baseline", type=click.Path(path_type=Path),
              help="Output directory for baseline results.")
def baseline(suite_path: Path, output_dir: Path):
    """Run all benchmark tasks with current config and report results."""
    from cc_optimize.adapter.config_reader import read_config
    from cc_optimize.adapter.task_runner import run_task
    from cc_optimize.benchmark.loader import load_suite
    from cc_optimize.evaluation.evaluator import evaluate_task_run

    suite = load_suite(suite_path)
    config = read_config(suite.repo_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Running baseline for suite: {suite.name}")
    click.echo(f"Tasks: {len(suite.tasks)}")
    click.echo()

    results = []
    for task in suite.tasks:
        click.echo(f"Running {task.id}...")
        run_result = run_task(task, config, output_dir)
        task_result = evaluate_task_run(task, config, run_result)
        results.append(task_result)
        status = "PASS" if task_result.success else "FAIL"
        click.echo(
            f"  {status} | efficiency={task_result.signals.efficiency_score:.2f} "
            f"| turns={task_result.signals.turn_count} "
            f"| tokens={task_result.tokens_total}"
        )

    passed = sum(1 for r in results if r.success)
    click.echo()
    click.echo(f"Results: {passed}/{len(results)} tasks passed")


@main.command()
@click.argument("suite_path", type=click.Path(exists=True, path_type=Path))
@click.option("--max-calls", "-n", default=150, type=int, help="Max GEPA metric calls.")
@click.option("--output-dir", "-o", default="optimization", type=click.Path(path_type=Path),
              help="Output directory for optimization results.")
@click.option("--config", "config_path", default=None, type=click.Path(path_type=Path),
              help="Path to optimization config YAML.")
def optimize(suite_path: Path, max_calls: int, output_dir: Path, config_path: Path | None):
    """Run GEPA optimization loop on Claude Code config."""
    from cc_optimize.benchmark.loader import load_suite
    from cc_optimize.config import GlobalConfig
    from cc_optimize.optimization.runner import run_optimization
    from cc_optimize.optimization.seed import generate_seed

    global_config = GlobalConfig.load(config_path)
    global_config.optimization.max_metric_calls = max_calls

    suite = load_suite(suite_path)
    seed_config = generate_seed(suite.repo_path)

    click.echo(f"Starting optimization for suite: {suite.name}")
    click.echo(f"Max metric calls: {max_calls}")
    click.echo()

    report = run_optimization(
        suite=suite,
        seed_config=seed_config,
        config=global_config.optimization,
        work_dir=output_dir / "work",
        early_stop_config=global_config.early_stopping,
    )

    report.save(output_dir)
    click.echo(f"Optimization complete. Report saved to {output_dir}")
    click.echo(f"Total metric calls: {report.total_metric_calls}")
    click.echo(f"Wall time: {report.wall_time_seconds:.1f}s")


@main.command("apply")
@click.argument("candidate_path", type=click.Path(exists=True, path_type=Path))
@click.argument("repo_path", type=click.Path(exists=True, path_type=Path))
def apply_cmd(candidate_path: Path, repo_path: Path):
    """Apply a saved candidate YAML to a repository."""
    from cc_optimize.adapter.config_applier import apply_config
    from cc_optimize.models.candidate import ConfigCandidate, RuleFile, SkillFile

    with open(candidate_path) as f:
        data = yaml.safe_load(f)

    candidate = ConfigCandidate(
        id=data.get("id", "applied"),
        claude_md=data.get("claude_md", ""),
        rules=[
            RuleFile(
                filename=r["filename"],
                content=r["content"],
                paths=r.get("paths"),
            )
            for r in data.get("rules", [])
        ],
        skills=[
            SkillFile(name=s["name"], skill_md=s["skill_md"])
            for s in data.get("skills", [])
        ],
        settings_json=data.get("settings_json", {}),
        context_imports=data.get("context_imports", []),
    )

    click.echo(f"Applying candidate {candidate.id} to {repo_path}")
    click.echo(f"  CLAUDE.md: {len(candidate.claude_md)} chars")
    click.echo(f"  Rules: {len(candidate.rules)}")
    click.echo(f"  Skills: {len(candidate.skills)}")

    if not click.confirm("Proceed?"):
        click.echo("Aborted.")
        return

    try:
        apply_config(candidate, repo_path)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    click.echo("Applied successfully.")


@main.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
def report(output_dir: Path):
    """Display a previously generated optimization report."""
    report_path = output_dir / "report.md"
    if not report_path.exists():
        click.echo(f"No report found at {report_path}")
        raise SystemExit(1)
    click.echo(report_path.read_text())
