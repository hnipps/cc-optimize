"""Verify ConfigCandidate round-trip invariant from spec Section 3.2.

Spec states: ConfigCandidate.from_component_dict(c.to_component_dict(), c.id,
c.settings_json) must produce a ConfigCandidate that, when applied to a repo,
produces identical files to applying c directly.
"""
from __future__ import annotations

import json
from pathlib import Path

from cc_optimize.adapter.config_applier import apply_config
from cc_optimize.models.candidate import ConfigCandidate, RuleFile, SkillFile


def _create_git_dir(path: Path) -> None:
    """Create a minimal .git dir so apply_config doesn't raise."""
    (path / ".git").mkdir(parents=True, exist_ok=True)


def _read_all_files(root: Path) -> dict[str, str]:
    """Read all files under root into a dict of relative_path -> content."""
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".git" not in p.parts:
            files[str(p.relative_to(root))] = p.read_text()
    return files


class TestRoundTrip:
    def test_full_candidate(self, tmp_path):
        """Full candidate with claude_md, 2 rules, 1 skill, context_imports, settings."""
        candidate = ConfigCandidate(
            id="test-001",
            claude_md="# Project\n@docs/api.md\n\nUse structured logging.",
            rules=[
                RuleFile(
                    filename="go-conventions.md",
                    content="Always handle errors explicitly.\nUse table-driven tests.",
                    paths=["internal/**/*.go", "cmd/**/*.go"],
                ),
                RuleFile(
                    filename="general.md",
                    content="Follow the existing code style.",
                    paths=None,  # unconditional
                ),
            ],
            skills=[
                SkillFile(
                    name="questdb-patterns",
                    skill_md="# QuestDB Patterns\nUse parameterized queries.",
                ),
            ],
            settings_json={"permissions": {"allow": ["Bash(*)"]}},
            context_imports=["docs/api.md"],
        )

        # Apply original
        dir_a = tmp_path / "repo_a"
        dir_a.mkdir()
        _create_git_dir(dir_a)
        apply_config(candidate, dir_a)

        # Round-trip through component dict
        comp_dict = candidate.to_component_dict()
        restored = ConfigCandidate.from_component_dict(
            comp_dict, candidate.id, candidate.settings_json
        )

        # Apply round-tripped
        dir_b = tmp_path / "repo_b"
        dir_b.mkdir()
        _create_git_dir(dir_b)
        apply_config(restored, dir_b)

        # Compare all files
        files_a = _read_all_files(dir_a)
        files_b = _read_all_files(dir_b)

        assert files_a.keys() == files_b.keys(), (
            f"File set mismatch: {files_a.keys() - files_b.keys()} vs {files_b.keys() - files_a.keys()}"
        )
        for path in files_a:
            assert files_a[path] == files_b[path], f"Content mismatch in {path}"

    def test_structural_equivalence(self, tmp_path):
        """Round-tripped candidate has matching fields."""
        candidate = ConfigCandidate(
            id="test-002",
            claude_md="Be concise.",
            rules=[
                RuleFile(filename="a.md", content="Rule A content", paths=["src/**"]),
                RuleFile(filename="b.md", content="Rule B content", paths=None),
            ],
            skills=[SkillFile(name="my-skill", skill_md="Skill instructions.")],
            settings_json={"key": "value"},
            context_imports=["path/to/doc.md"],
        )

        comp_dict = candidate.to_component_dict()
        restored = ConfigCandidate.from_component_dict(
            comp_dict, candidate.id, candidate.settings_json
        )

        assert restored.id == candidate.id
        assert restored.claude_md == candidate.claude_md
        assert restored.settings_json == candidate.settings_json
        assert restored.context_imports == candidate.context_imports

        # Rules match (order may differ due to dict iteration)
        orig_rules = {r.filename: r for r in candidate.rules}
        rest_rules = {r.filename: r for r in restored.rules}
        assert orig_rules.keys() == rest_rules.keys()
        for fname in orig_rules:
            assert orig_rules[fname].content == rest_rules[fname].content
            assert orig_rules[fname].paths == rest_rules[fname].paths

        # Skills match
        orig_skills = {s.name: s for s in candidate.skills}
        rest_skills = {s.name: s for s in restored.skills}
        assert orig_skills.keys() == rest_skills.keys()
        for name in orig_skills:
            assert orig_skills[name].skill_md == rest_skills[name].skill_md


class TestEdgeCases:
    def test_empty_claude_md(self, tmp_path):
        candidate = ConfigCandidate(id="e1", claude_md="", settings_json={})
        comp_dict = candidate.to_component_dict()
        restored = ConfigCandidate.from_component_dict(comp_dict, "e1", {})
        assert restored.claude_md == ""

    def test_no_rules(self, tmp_path):
        candidate = ConfigCandidate(id="e2", claude_md="hello", rules=[])
        comp_dict = candidate.to_component_dict()
        restored = ConfigCandidate.from_component_dict(comp_dict, "e2")
        assert restored.rules == []

    def test_no_skills(self, tmp_path):
        candidate = ConfigCandidate(id="e3", claude_md="hello", skills=[])
        comp_dict = candidate.to_component_dict()
        restored = ConfigCandidate.from_component_dict(comp_dict, "e3")
        assert restored.skills == []

    def test_path_scope_multiple_paths(self, tmp_path):
        """PATH_SCOPE with multiple paths survives round-trip."""
        candidate = ConfigCandidate(
            id="e4",
            claude_md="x",
            rules=[
                RuleFile(
                    filename="multi.md",
                    content="Multi-path rule.",
                    paths=["src/**/*.go", "internal/**/*.go", "cmd/**"],
                ),
            ],
        )
        comp_dict = candidate.to_component_dict()
        restored = ConfigCandidate.from_component_dict(comp_dict, "e4")
        assert len(restored.rules) == 1
        assert restored.rules[0].paths == ["src/**/*.go", "internal/**/*.go", "cmd/**"]

    def test_special_characters_in_rule_content(self, tmp_path):
        """Special chars (quotes, newlines, unicode) in rule content survive."""
        content = 'Use "structured" errors.\nDon\'t panic.\n\u2603 snowman.'
        candidate = ConfigCandidate(
            id="e5",
            claude_md="x",
            rules=[RuleFile(filename="special.md", content=content, paths=None)],
        )
        comp_dict = candidate.to_component_dict()
        restored = ConfigCandidate.from_component_dict(comp_dict, "e5")
        assert restored.rules[0].content == content

    def test_file_identity_with_empty_components(self, tmp_path):
        """Apply empty-ish candidate to two dirs, files are identical."""
        candidate = ConfigCandidate(id="min", claude_md="", rules=[], skills=[], settings_json={})

        dir_a = tmp_path / "a"
        dir_a.mkdir()
        _create_git_dir(dir_a)
        apply_config(candidate, dir_a)

        comp_dict = candidate.to_component_dict()
        restored = ConfigCandidate.from_component_dict(comp_dict, "min", {})

        dir_b = tmp_path / "b"
        dir_b.mkdir()
        _create_git_dir(dir_b)
        apply_config(restored, dir_b)

        assert _read_all_files(dir_a) == _read_all_files(dir_b)
