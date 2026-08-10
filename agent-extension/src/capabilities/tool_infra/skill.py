"""Skill — 从 skills/ 目录加载 SKILL.md，提示词注入模式。

仿照 Claude Code:
  - skills/<skill-name>/SKILL.md 定义 Skill（YAML frontmatter + Markdown body）
  - 启动时扫描目录，提取 name + description + path
  - Skills 列表注入 system prompt（始终全量可见）
  - 唯一的 Skill 工具注册到 ToolRegistry，调用时从磁盘读 body 返回
"""

import os
import re
import yaml

_SKILLS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "skills",
)


class SkillDef:
    """Skill 元信息，不持有 body（调用时从磁盘读）。"""

    def __init__(self, name: str, description: str, path: str):
        self.name = name
        self.description = description
        self.path = path


def scan_skills() -> list[SkillDef]:
    """扫描 skills/ 目录，返回所有 Skill 的元信息。

    只读 frontmatter（name + description），不加载 body。
    """
    skills: list[SkillDef] = []
    if not os.path.isdir(_SKILLS_DIR):
        return skills

    for entry in sorted(os.listdir(_SKILLS_DIR)):
        skill_dir = os.path.join(_SKILLS_DIR, entry)
        if not os.path.isdir(skill_dir):
            continue

        md_path = os.path.join(skill_dir, "SKILL.md")
        if not os.path.isfile(md_path):
            continue

        # 只解析 frontmatter，不加载整个 body
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        match = re.match(
            r"^---\s*\n(.*?)\n---", content, re.DOTALL,
        )
        if not match:
            continue

        frontmatter = yaml.safe_load(match.group(1))
        name = frontmatter.get("name", "")
        description = frontmatter.get("description", "")
        if not name or not description:
            continue

        skills.append(SkillDef(name=name, description=description, path=md_path))

    return skills


def load_skill_prompt(name: str) -> str | None:
    """根据 Skill 名从磁盘读 SKILL.md，返回 body（去掉 frontmatter）。

    每次调用都重新读取，修改 SKILL.md 后立即生效。
    """
    full_path = os.path.join(_SKILLS_DIR, name, "SKILL.md")
    if not os.path.isfile(full_path):
        return None

    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.match(
        r"^---\s*\n.*?\n---\s*\n(.*)", content, re.DOTALL,
    )
    if not match:
        return content.strip()

    return match.group(1).strip()
