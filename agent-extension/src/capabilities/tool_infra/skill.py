"""Skill — 从 skills/ 目录加载 SKILL.md，提示词注入模式。

仿照 Claude Code:
  - skills/<skill-name>/SKILL.md 定义 Skill（YAML frontmatter + Markdown body）
  - scan_skills() 扫描目录，返回 [{name, description}, ...]（给 core.py 注入 system prompt）
  - get_skill_tool() 返回 Skill 入口工具的注册数据（给 ToolRegistry 调用）
  - fn 闭包调用时从磁盘读 body（hot-reload）
"""

import os
import re
import yaml

_SKILLS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "skills",
)

# ----------------------------------------------------------------
# 内部
# ----------------------------------------------------------------


def _parse_skill_md(path: str) -> dict | None:
    """一次读文件、一次正则，返回 {name, description, body}。

    SKILL.md 格式:
        ---
        name: skill-name
        description: 触发条件
        ---
        # 操作指南
        ...
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return None

    match = re.match(
        r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL,
    )
    if not match:
        return None

    frontmatter = yaml.safe_load(match.group(1))
    name = frontmatter.get("name", "")
    description = frontmatter.get("description", "")
    body = match.group(2).strip()

    if not name or not description:
        return None

    return {"name": name, "description": description, "body": body}


# ----------------------------------------------------------------
# 公开
# ----------------------------------------------------------------


def scan_skills() -> list[dict]:
    """扫描 skills/ 目录，返回 [{name, description}, ...]。

    只解析 frontmatter，不返回 body。给 core.py 注入 system prompt 用。
    """
    if not os.path.isdir(_SKILLS_DIR):
        return []

    skills: list[dict] = []
    for entry in sorted(os.listdir(_SKILLS_DIR)):
        skill_dir = os.path.join(_SKILLS_DIR, entry)
        if not os.path.isdir(skill_dir):
            continue
        md_path = os.path.join(skill_dir, "SKILL.md")
        if not os.path.isfile(md_path):
            continue
        parsed = _parse_skill_md(md_path)
        if parsed:
            skills.append({
                "name": parsed["name"],
                "description": parsed["description"],
            })
    return skills


def get_skill_tool() -> dict | None:
    """返回 Skill 入口工具的注册数据 {name, description, parameters, fn}。

    fn(name) 每次从磁盘读 SKILL.md body（hot-reload）。
    无 Skill 时返回 None。
    """
    skills = scan_skills()
    if not skills:
        return None

    skill_names = [s["name"] for s in skills]
    skill_desc_lines = [
        f"- {s['name']}: {s['description']}" for s in skills
    ]

    def fn(name: str) -> str:
        full_path = os.path.join(_SKILLS_DIR, name, "SKILL.md")
        if not os.path.isfile(full_path):
            return f"Skill '{name}' 不存在。可用 Skills: {', '.join(skill_names)}"
        parsed = _parse_skill_md(full_path)
        if parsed is None:
            return f"Skill '{name}' 解析失败。"
        return parsed["body"]

    return {
        "name": "Skill",
        "description": (
            "加载一个 Skill 的操作指南。"
            "可用 Skills:\n" + "\n".join(skill_desc_lines)
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "要加载的 Skill 名称",
                },
            },
            "required": ["name"],
        },
        "fn": fn,
    }
