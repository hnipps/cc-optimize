from __future__ import annotations

import json
from pathlib import Path

import pytest

from cc_optimize.adapter.config_applier import apply_config
from cc_optimize.adapter.config_reader import read_config

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "sample_config"


class TestApplyConfig:
    def test_round_trip(self, tmp_path: Path) -> None:
        """Read from fixture, apply to tmp dir, read back and verify."""
        (tmp_path / ".git").mkdir()
        original = read_config(FIXTURE_DIR)
        apply_config(original, tmp_path)
        restored = read_config(tmp_path)

        assert restored.claude_md == original.claude_md
        assert len(restored.rules) == len(original.rules)
        for orig_rule in original.rules:
            rest_rule = next(r for r in restored.rules if r.filename == orig_rule.filename)
            assert rest_rule.content == orig_rule.content
            assert rest_rule.paths == orig_rule.paths
        assert len(restored.skills) == len(original.skills)
        for orig_skill in original.skills:
            rest_skill = next(s for s in restored.skills if s.name == orig_skill.name)
            assert rest_skill.skill_md == orig_skill.skill_md
        assert restored.context_imports == original.context_imports

    def test_frontmatter_with_paths_written_correctly(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        original = read_config(FIXTURE_DIR)
        apply_config(original, tmp_path)

        rule_file = tmp_path / ".claude" / "rules" / "go-conventions.md"
        content = rule_file.read_text()
        assert content.startswith("---\n")
        assert '"*.go"' in content
        assert '"internal/**"' in content

    def test_rules_without_paths_have_no_frontmatter(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        original = read_config(FIXTURE_DIR)
        apply_config(original, tmp_path)

        rule_file = tmp_path / ".claude" / "rules" / "general.md"
        content = rule_file.read_text()
        assert not content.startswith("---")

    def test_error_on_non_git_directory(self, tmp_path: Path) -> None:
        original = read_config(FIXTURE_DIR)
        with pytest.raises(ValueError, match="No .git"):
            apply_config(original, tmp_path)

    def test_skills_directory_created_with_skill_md(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        original = read_config(FIXTURE_DIR)
        apply_config(original, tmp_path)

        skill_md = tmp_path / ".claude" / "skills" / "questdb-patterns" / "SKILL.md"
        assert skill_md.exists()
        assert "QuestDB" in skill_md.read_text()

    def test_settings_json_written(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        original = read_config(FIXTURE_DIR)
        apply_config(original, tmp_path)

        settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert settings == {"permissions": {"allow": ["Bash(*)"]}}
