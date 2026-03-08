from __future__ import annotations

import json
import shutil
from pathlib import Path

from cc_optimize.models.candidate import ConfigCandidate


def _generate_frontmatter(paths: list[str]) -> str:
    lines = ["---"]
    lines.append("paths:")
    for p in paths:
        lines.append(f'  - "{p}"')
    lines.append("---")
    return "\n".join(lines) + "\n"


def apply_config(candidate: ConfigCandidate, repo_path: Path) -> None:
    """
    1. Verify repo_path exists and contains .git/ (or .git file for worktrees)
    2. Write {repo_path}/CLAUDE.md
    3. Delete all .md files in {repo_path}/.claude/rules/ (create dir if needed)
    4. Write each RuleFile:
       - If rule.paths is not None: prepend YAML frontmatter with paths
       - Write to {repo_path}/.claude/rules/{rule.filename}
    5. Delete all subdirectories in {repo_path}/.claude/skills/ (create dir if needed)
    6. For each SkillFile:
       - Create {repo_path}/.claude/skills/{skill.name}/
       - Write SKILL.md
    7. Write {repo_path}/.claude/settings.json

    Raises ValueError if repo_path has no .git directory/file.
    """
    # 1. Verify git
    git_path = repo_path / ".git"
    if not git_path.exists():
        raise ValueError(f"No .git directory or file found at {repo_path}")

    # 2. Write CLAUDE.md
    (repo_path / "CLAUDE.md").write_text(candidate.claude_md)

    # 3. Clean and recreate rules dir
    rules_dir = repo_path / ".claude" / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    for md_file in rules_dir.glob("*.md"):
        md_file.unlink()

    # 4. Write rules
    for rule in candidate.rules:
        rule_path = rules_dir / rule.filename
        content = ""
        if rule.paths is not None:
            content = _generate_frontmatter(rule.paths)
        content += rule.content
        rule_path.write_text(content)

    # 5. Clean and recreate skills dir
    skills_dir = repo_path / ".claude" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    for subdir in skills_dir.iterdir():
        if subdir.is_dir():
            shutil.rmtree(subdir)

    # 6. Write skills
    for skill in candidate.skills:
        skill_dir = skills_dir / skill.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(skill.skill_md)

    # 7. Write settings.json
    settings_path = repo_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(candidate.settings_json, indent=2) + "\n")
