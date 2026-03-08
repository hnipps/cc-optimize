from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import yaml

from cc_optimize.models.candidate import ConfigCandidate, RuleFile, SkillFile


def read_config(repo_path: Path) -> ConfigCandidate:
    r"""
    1. Read {repo_path}/CLAUDE.md -> claude_md (empty string if missing)
    2. Read all .md files in {repo_path}/.claude/rules/
       - Parse YAML frontmatter for paths: field
       - Frontmatter delimited by --- on first line and --- on a subsequent line
       - Files without frontmatter or without paths: key -> unconditional (paths=None)
    3. Read all subdirectories in {repo_path}/.claude/skills/
       - Each must contain a SKILL.md -> skill_md
       - Directory name -> skill name
    4. Read {repo_path}/.claude/settings.json -> settings_json (empty dict if missing)
    5. Extract @import references from claude_md content
       - Pattern: lines matching ^@(\S+) -> capture the path
    6. Generate UUID for candidate id
    7. Return ConfigCandidate with parent_id=None
    """
    # 1. Read CLAUDE.md
    claude_md_path = repo_path / "CLAUDE.md"
    if claude_md_path.exists():
        claude_md = claude_md_path.read_text()
    else:
        claude_md = ""

    # 2. Read rules
    rules: list[RuleFile] = []
    rules_dir = repo_path / ".claude" / "rules"
    if rules_dir.is_dir():
        for md_file in sorted(rules_dir.glob("*.md")):
            content = md_file.read_text()
            paths = None
            body = content

            # Parse frontmatter
            lines = content.split("\n")
            if lines and lines[0].strip() == "---":
                # Find closing ---
                end_idx = None
                for i in range(1, len(lines)):
                    if lines[i].strip() == "---":
                        end_idx = i
                        break
                if end_idx is not None:
                    frontmatter_text = "\n".join(lines[1:end_idx])
                    body = "\n".join(lines[end_idx + 1:])
                    # Strip leading newline from body
                    if body.startswith("\n"):
                        body = body[1:]
                    parsed = yaml.safe_load(frontmatter_text)
                    if isinstance(parsed, dict) and "paths" in parsed:
                        paths = parsed["paths"]

            rules.append(RuleFile(filename=md_file.name, content=body, paths=paths))

    # 3. Read skills
    skills: list[SkillFile] = []
    skills_dir = repo_path / ".claude" / "skills"
    if skills_dir.is_dir():
        for skill_subdir in sorted(skills_dir.iterdir()):
            if skill_subdir.is_dir():
                skill_md_path = skill_subdir / "SKILL.md"
                if skill_md_path.exists():
                    skills.append(
                        SkillFile(
                            name=skill_subdir.name,
                            skill_md=skill_md_path.read_text(),
                        )
                    )

    # 4. Read settings.json
    settings_path = repo_path / ".claude" / "settings.json"
    if settings_path.exists():
        settings_json = json.loads(settings_path.read_text())
    else:
        settings_json = {}

    # 5. Extract @import references
    context_imports: list[str] = []
    for line in claude_md.split("\n"):
        match = re.match(r"^@(\S+)", line)
        if match:
            context_imports.append(match.group(1))

    # 6 & 7. Build and return
    return ConfigCandidate(
        id=str(uuid.uuid4()),
        claude_md=claude_md,
        rules=rules,
        skills=skills,
        settings_json=settings_json,
        context_imports=context_imports,
        parent_id=None,
    )
