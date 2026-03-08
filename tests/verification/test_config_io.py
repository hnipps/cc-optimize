"""Verify config reader/applier round-trip: read_config -> apply_config -> read_config.

Creates a tmp dir mimicking a real repo with CLAUDE.md, rules, skills, and settings.
"""
from __future__ import annotations

import json
from pathlib import Path

from cc_optimize.adapter.config_applier import apply_config
from cc_optimize.adapter.config_reader import read_config


def _create_repo(path: Path) -> None:
    """Create a mock repo with all Claude Code config files."""
    (path / ".git").mkdir(parents=True)

    # CLAUDE.md with @import
    (path / "CLAUDE.md").write_text(
        "@docs/architecture.md\n\n# Project Guidelines\nUse structured error responses.\n"
    )

    # Rules dir with 2 rules
    rules_dir = path / ".claude" / "rules"
    rules_dir.mkdir(parents=True)

    # Rule with frontmatter paths
    (rules_dir / "go-conventions.md").write_text(
        '---\npaths:\n  - "internal/**/*.go"\n  - "cmd/**/*.go"\n---\n'
        "Always handle errors with wrapped context.\nUse table-driven tests.\n"
    )

    # Rule without frontmatter (unconditional)
    (rules_dir / "general.md").write_text(
        "Follow existing code style.\nKeep functions under 50 lines.\n"
    )

    # Skills dir with 1 skill
    skill_dir = path / ".claude" / "skills" / "db-patterns"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# Database Patterns\nUse parameterized queries.\nAvoid N+1 queries.\n"
    )

    # Settings
    (path / ".claude" / "settings.json").write_text(
        json.dumps({"permissions": {"allow": ["Bash(*)"]}}, indent=2) + "\n"
    )


class TestReadApplyReadRoundTrip:
    def test_round_trip(self, tmp_path):
        # Create source repo
        src = tmp_path / "source"
        src.mkdir()
        _create_repo(src)

        # Read config from source
        config_a = read_config(src)

        # Apply to a different directory
        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / ".git").mkdir()
        apply_config(config_a, dest)

        # Read config from destination
        config_b = read_config(dest)

        # Assert structural equivalence
        assert config_b.claude_md == config_a.claude_md

        # Rules
        rules_a = {r.filename: r for r in config_a.rules}
        rules_b = {r.filename: r for r in config_b.rules}
        assert rules_a.keys() == rules_b.keys()
        for fname in rules_a:
            assert rules_a[fname].content == rules_b[fname].content, f"Rule {fname} content mismatch"
            assert rules_a[fname].paths == rules_b[fname].paths, f"Rule {fname} paths mismatch"

        # Skills
        skills_a = {s.name: s for s in config_a.skills}
        skills_b = {s.name: s for s in config_b.skills}
        assert skills_a.keys() == skills_b.keys()
        for name in skills_a:
            assert skills_a[name].skill_md == skills_b[name].skill_md, f"Skill {name} mismatch"

        # Settings
        assert config_b.settings_json == config_a.settings_json

        # Context imports
        assert config_b.context_imports == config_a.context_imports

    def test_round_trip_preserves_rule_paths(self, tmp_path):
        """Frontmatter paths survive read -> apply -> read."""
        src = tmp_path / "src"
        src.mkdir()
        _create_repo(src)

        config_a = read_config(src)
        go_rule = next(r for r in config_a.rules if r.filename == "go-conventions.md")
        assert go_rule.paths is not None
        assert "internal/**/*.go" in go_rule.paths
        assert "cmd/**/*.go" in go_rule.paths

        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / ".git").mkdir()
        apply_config(config_a, dest)

        config_b = read_config(dest)
        go_rule_b = next(r for r in config_b.rules if r.filename == "go-conventions.md")
        assert go_rule_b.paths == go_rule.paths

    def test_empty_repo(self, tmp_path):
        """Repo with no Claude Code config files produces empty candidate."""
        repo = tmp_path / "empty_repo"
        repo.mkdir()
        (repo / ".git").mkdir()

        config = read_config(repo)
        assert config.claude_md == ""
        assert config.rules == []
        assert config.skills == []
        assert config.settings_json == {}
        assert config.context_imports == []

    def test_unconditional_rule_has_no_paths(self, tmp_path):
        """Rule without frontmatter has paths=None."""
        src = tmp_path / "src"
        src.mkdir()
        _create_repo(src)

        config = read_config(src)
        general_rule = next(r for r in config.rules if r.filename == "general.md")
        assert general_rule.paths is None
