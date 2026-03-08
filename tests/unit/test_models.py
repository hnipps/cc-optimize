from __future__ import annotations

from cc_optimize.models.candidate import ConfigCandidate, RuleFile, SkillFile
from cc_optimize.models.result import CompositeScore


class TestConfigCandidateRoundTrip:
    def test_full_round_trip(self):
        original = ConfigCandidate(
            id="test-1",
            claude_md="# Project\nUse Go conventions.",
            rules=[
                RuleFile(
                    filename="go-conventions.md",
                    content="Always use gofmt.",
                    paths=["*.go", "internal/**"],
                ),
                RuleFile(
                    filename="general.md",
                    content="Be concise.",
                    paths=None,
                ),
            ],
            skills=[
                SkillFile(name="questdb-patterns", skill_md="Use QuestDB for time series."),
            ],
            settings_json={"permissions": {"allow": ["Bash(*)"]}},
            context_imports=["docs/architecture.md", "docs/api.md"],
        )
        components = original.to_component_dict()
        restored = ConfigCandidate.from_component_dict(
            components, original.id, original.settings_json
        )
        assert restored.claude_md == original.claude_md
        assert len(restored.rules) == len(original.rules)
        for orig_rule in original.rules:
            matching = [r for r in restored.rules if r.filename == orig_rule.filename]
            assert len(matching) == 1
            assert matching[0].content == orig_rule.content
            assert matching[0].paths == orig_rule.paths
        assert len(restored.skills) == len(original.skills)
        assert restored.skills[0].name == original.skills[0].name
        assert restored.skills[0].skill_md == original.skills[0].skill_md
        assert restored.settings_json == original.settings_json
        assert restored.context_imports == original.context_imports

    def test_empty_candidate(self):
        original = ConfigCandidate(id="empty", claude_md="")
        components = original.to_component_dict()
        assert components == {"claude_md": ""}
        restored = ConfigCandidate.from_component_dict(components, "empty")
        assert restored.claude_md == ""
        assert restored.rules == []
        assert restored.skills == []
        assert restored.context_imports == []

    def test_path_scope_unconditional(self):
        candidate = ConfigCandidate(
            id="t",
            claude_md="",
            rules=[RuleFile(filename="test.md", content="content", paths=None)],
        )
        components = candidate.to_component_dict()
        assert "unconditional" in components["rule:test.md"]
        restored = ConfigCandidate.from_component_dict(components, "t")
        assert restored.rules[0].paths is None
        assert restored.rules[0].content == "content"

    def test_path_scope_with_paths(self):
        candidate = ConfigCandidate(
            id="t",
            claude_md="",
            rules=[RuleFile(filename="go.md", content="go stuff", paths=["*.go"])],
        )
        components = candidate.to_component_dict()
        assert '["*.go"]' in components["rule:go.md"]
        restored = ConfigCandidate.from_component_dict(components, "t")
        assert restored.rules[0].paths == ["*.go"]


class TestCompositeScore:
    def test_weighted_scalar_defaults(self):
        score = CompositeScore(correctness=1.0, efficiency=1.0, conversation_quality=1.0)
        assert score.weighted_scalar() == 1.0

    def test_weighted_scalar_custom(self):
        score = CompositeScore(correctness=0.8, efficiency=0.6, conversation_quality=0.4)
        result = score.weighted_scalar((0.5, 0.25, 0.25))
        expected = 0.5 * 0.8 + 0.25 * 0.6 + 0.25 * 0.4
        assert abs(result - expected) < 1e-9
