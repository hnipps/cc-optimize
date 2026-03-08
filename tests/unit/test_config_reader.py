from __future__ import annotations

from pathlib import Path

from cc_optimize.adapter.config_reader import read_config

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "sample_config"


class TestReadConfig:
    def test_reads_claude_md(self) -> None:
        candidate = read_config(FIXTURE_DIR)
        assert "# Project Config" in candidate.claude_md
        assert "Use Go conventions." in candidate.claude_md

    def test_reads_rules_with_paths(self) -> None:
        candidate = read_config(FIXTURE_DIR)
        go_rule = next(r for r in candidate.rules if r.filename == "go-conventions.md")
        assert go_rule.paths == ["*.go", "internal/**"]
        assert "Always use gofmt." in go_rule.content
        # Content should not contain frontmatter
        assert "---" not in go_rule.content

    def test_reads_rules_without_paths(self) -> None:
        candidate = read_config(FIXTURE_DIR)
        general_rule = next(r for r in candidate.rules if r.filename == "general.md")
        assert general_rule.paths is None
        assert "Be concise in responses." in general_rule.content

    def test_reads_skills(self) -> None:
        candidate = read_config(FIXTURE_DIR)
        assert len(candidate.skills) == 1
        assert candidate.skills[0].name == "questdb-patterns"
        assert "QuestDB" in candidate.skills[0].skill_md

    def test_reads_settings_json(self) -> None:
        candidate = read_config(FIXTURE_DIR)
        assert candidate.settings_json == {"permissions": {"allow": ["Bash(*)"]}}

    def test_extracts_context_imports(self) -> None:
        candidate = read_config(FIXTURE_DIR)
        assert "docs/architecture.md" in candidate.context_imports

    def test_has_uuid_and_no_parent(self) -> None:
        candidate = read_config(FIXTURE_DIR)
        assert candidate.id  # non-empty
        assert candidate.parent_id is None

    def test_missing_claude_md_returns_empty_string(self, tmp_path: Path) -> None:
        # No CLAUDE.md in tmp_path
        candidate = read_config(tmp_path)
        assert candidate.claude_md == ""

    def test_missing_rules_dir_returns_empty_list(self, tmp_path: Path) -> None:
        candidate = read_config(tmp_path)
        assert candidate.rules == []

    def test_missing_skills_dir_returns_empty_list(self, tmp_path: Path) -> None:
        candidate = read_config(tmp_path)
        assert candidate.skills == []
