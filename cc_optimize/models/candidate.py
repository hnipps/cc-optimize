from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class RuleFile:
    filename: str
    content: str
    paths: list[str] | None = None


@dataclass
class SkillFile:
    name: str
    skill_md: str


@dataclass
class ConfigCandidate:
    id: str
    claude_md: str
    rules: list[RuleFile] = field(default_factory=list)
    skills: list[SkillFile] = field(default_factory=list)
    settings_json: dict = field(default_factory=dict)
    context_imports: list[str] = field(default_factory=list)
    parent_id: str | None = None
    mutation_rationale: str = ""

    def to_component_dict(self) -> dict[str, str]:
        components: dict[str, str] = {}
        components["claude_md"] = self.claude_md
        for rule in self.rules:
            key = f"rule:{rule.filename}"
            if rule.paths:
                header = f"# PATH_SCOPE: {json.dumps(rule.paths)}\n"
            else:
                header = "# PATH_SCOPE: unconditional\n"
            components[key] = header + rule.content
        for skill in self.skills:
            components[f"skill:{skill.name}"] = skill.skill_md
        if self.context_imports:
            components["context_imports"] = "\n".join(self.context_imports)
        return components

    @classmethod
    def from_component_dict(
        cls,
        components: dict[str, str],
        candidate_id: str,
        settings_json: dict | None = None,
    ) -> ConfigCandidate:
        claude_md = components.get("claude_md", "")
        rules: list[RuleFile] = []
        skills: list[SkillFile] = []
        context_imports: list[str] = []
        for key, value in components.items():
            if key.startswith("rule:"):
                filename = key[len("rule:"):]
                lines = value.split("\n", 1)
                paths = None
                content = value
                if lines[0].startswith("# PATH_SCOPE:"):
                    scope_str = lines[0][len("# PATH_SCOPE:"):].strip()
                    if scope_str != "unconditional":
                        paths = json.loads(scope_str)
                    content = lines[1] if len(lines) > 1 else ""
                rules.append(RuleFile(filename=filename, content=content, paths=paths))
            elif key.startswith("skill:"):
                name = key[len("skill:"):]
                skills.append(SkillFile(name=name, skill_md=value))
            elif key == "context_imports":
                context_imports = [
                    line.strip() for line in value.strip().split("\n") if line.strip()
                ]
        return cls(
            id=candidate_id,
            claude_md=claude_md,
            rules=rules,
            skills=skills,
            settings_json=settings_json or {},
            context_imports=context_imports,
        )
