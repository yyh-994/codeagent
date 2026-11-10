"""Skills system — discover, parse, and execute built-in .claude/skills/*/SKILL.md.

技能系统：从 claude-code-agent 包内置的 .claude/skills/ 目录扫描 SKILL.md 文件，
解析每个 skill 的 frontmatter 元数据和 prompt 模板，向外提供 skill 发现、
查找、执行和描述生成等功能。

调用链（自上而下）：
  tools.py           → build_skill_descriptions() → 生成 skill 工具描述中的技能列表
  tools.py           → execute_skill()            → LLM 通过 skill tool 调用时执行

  skills.py 内部      → _discover_skills()          → 扫描并缓存所有内置 skill 定义
                      → _get_skill_by_name()        → 按名称查找单个 skill
                      → _resolve_skill_prompt()     → 替换模板中的占位符
                      → _load_skills_from_dir()    → 从指定目录加载 skill
                      → _parse_skill_file()        → 解析单个 SKILL.md 文件
"""

from __future__ import annotations  # 允许在类型注解中使用尚未定义的类名（如 SkillDefinition 自引用）

from dataclasses import dataclass  # 用数据类定义 SkillDefinition，自动生成 __init__
from pathlib import Path  # 跨平台路径操作

from .frontmatter import parse_frontmatter  # 解析 YAML frontmatter + Markdown body


@dataclass  # 标记为数据类，自动生成 __init__、__repr__、__eq__ 等方法
class SkillDefinition:
    """一个 Skill 的完整定义，包含元数据和渲染后的 prompt 模板。

    对应磁盘上 .claude/skills/<name>/SKILL.md 文件的解析结果。
    """

    name: str  # skill 名称，用于 /<name> 调用，取自 frontmatter 的 name 字段或目录名
    description: str  # skill 的一行描述，展示在 skill 列表中
    prompt_template: str = ""  # SKILL.md 的 body 部分，即去掉 frontmatter 后的 Markdown 模板

    when_to_use: str | None = None  # 可选的使用时机说明，帮助模型判断何时自动调用此 skill



# 模块级缓存：首次调用 _discover_skills() 后缓存结果，避免重复扫描文件系统
_cached_skills: list[SkillDefinition] | None = None

# ==================== 对外公开的 API ====================
def execute_skill(skill_name: str, args: str) -> dict | None:
    """执行一个 skill：按名称查找定义，渲染模板，返回执行所需的信息。

    Skill 始终 inline 执行：只返回渲染后的流程模板，由主 agent 照着步骤继续。
    需要独立上下文/角色的任务走 agent 工具（subagent.py）。

    调用方：
      - tools.py: _execute_skill() —— 当 LLM 通过 skill tool 调用某个 skill 时被触发

    Args:
        skill_name: skill 名称（如 "commit"、"remember"）
        args: 传入 skill 模板的参数字符串（替换 $ARGUMENTS）

    Returns:
        dict 包含 name、prompt（已渲染）；如果 skill 不存在则返回 None。
    """
    skill = _get_skill_by_name(skill_name)  # 按名称查找 skill 定义
    if not skill:  # 未找到对应 skill，无法执行
        return None  # 返回 None 通知调用方 skill 不存在
    return {  # 返回执行所需的全部信息
        "name": skill.name,  # skill 名称
        "prompt": _resolve_skill_prompt(skill, args),  # 渲染模板：将 args 填充到 prompt 中
    }
def _get_skill_by_name(name: str) -> SkillDefinition | None:
    """根据名称查找单个 skill 定义（支持带或不带前导 "/" 的名称）。

    调用方：
      - execute_skill() —— 用户通过 tool 调用 /技能名 时，据此找到对应的 skill 定义

    Args:
        name: skill 名称，如 "/remember" 或 "remember"

    Returns:
        匹配的 SkillDefinition，找不到则返回 None
    """
    clean = name.lstrip("/")  # 去掉前导 "/"，统一名称格式（/remember → remember）
    for s in _discover_skills():  # 遍历所有已发现的 skill（首次调用会触发扫描+缓存）
        if s.name == clean:  # 按名称精确匹配
            return s  # 找到即返回
    return None  # 遍历完毕仍未找到，返回 None

def _resolve_skill_prompt(skill: SkillDefinition, args: str) -> str:
    """将 skill 的 prompt 模板中的占位符替换为实际参数。

    调用方：
      - execute_skill() —— 执行 skill 时，用此函数将用户传入的 args 渲染到模板中

    支持的占位符：
      - $ARGUMENTS 或 ${ARGUMENTS} → 用户传入的参数文本

    Args:
        skill: skill 定义对象，包含 prompt_template
        args: 用户调用 /skill 时传入的参数字符串（可能为空）

    Returns:
        替换占位符后的最终 prompt 文本
    """
    import re  # 正则模块，按需导入（仅在执行 skill 时才需要）
    prompt = skill.prompt_template  # 获取模板副本（原始模板来自 SKILL.md 的 body 部分）
    prompt = re.sub(r"\$ARGUMENTS|\$\{ARGUMENTS\}", args, prompt)  # 将 $ARGUMENTS 或 ${ARGUMENTS} 替换为用户参数

    return prompt  # 返回替换完成后的最终 prompt



# 从技能头部元数据生成工具描述
def build_skill_descriptions() -> str:
    """从技能头部元数据生成 skill 工具的简短描述，不包含正文和参数说明。"""
    skills = _discover_skills()
    if not skills:
        return "获取并执行技能流程。当前没有可用技能。"
    lines = ["获取并执行技能流程。可用技能："]
    for skill in skills:
        description = " ".join(skill.description.split())
        when = " ".join(skill.when_to_use.split()) if skill.when_to_use else ""
        details = f"{description} 使用时机：{when}" if when else description
        lines.append(f"- {skill.name}：{details}")
    return "\n".join(lines)

# ==================== 内部辅助函数 ====================

def _discover_skills() -> list[SkillDefinition]:
    """扫描 claude-code-agent 包内置的 .claude/skills/，返回SkillDefinition列表。
    """
    global _cached_skills  # 声明要修改模块级缓存变量
    if _cached_skills is not None:  # 如果缓存已存在，直接返回，避免重复扫描磁盘
        return _cached_skills
    builtin_dir = Path(__file__).parent.parent / ".claude" / "skills"
    skills: dict[str, SkillDefinition] = {}  # 用 dict 收集，同名 skill 后者覆盖前者
    _load_skills_from_dir(builtin_dir, skills)  # 传入skill包路径，加载skill包路径下的md文件，并转为SkillDefinition存入skills列表
    _cached_skills = list(skills.values())  # 将 dict 的 values 转为 list 存入缓存
    return _cached_skills  # 返回SkillDefinition列表


def _load_skills_from_dir(base_dir: Path, skills: dict[str, SkillDefinition]) -> None:
    """从指定目录中加载所有 SKILL.md 文件，解析后追加到 skills 字典中。

    Args:
        base_dir: 要扫描的根目录
        skills: 用于收集结果的字典（就地修改），key 为 skill 名称，value 为 SkillDefinition
    """
    if not base_dir.is_dir():  # 目录不存在，跳过扫描
        return  # 静默返回，不做任何操作
    for entry in base_dir.iterdir():  # 遍历根目录下的所有条目
        if not entry.is_dir():  # 跳过文件，只处理子目录（每个 skill 一个子目录）
            continue  # 进入下一轮循环
        skill_file = entry / "SKILL.md"  # 构造 SKILL.md 的完整路径：<skill-dir>/SKILL.md
        if not skill_file.exists():  # 该子目录中没有 SKILL.md，不是有效的 skill 目录
            continue  # 跳过，进入下一轮循环
        skill = _parse_skill_file(skill_file)  # 解析 SKILL.md 文件，返回 SkillDefinition
        if skill:  # 解析成功（_parse_skill_file 在异常时返回 None）
            skills[skill.name] = skill  # 以名称为 key 存入字典


def _parse_skill_file(file_path: Path) -> SkillDefinition | None:
    """解析单个 SKILL.md 文件，从 frontmatter 和 body 中提取 skill 定义。
    SKILL.md 文件格式：
      ---
      name: skill-name           ← YAML frontmatter 元数据
      description: 一行描述
      when-to-use:
      ---
      ← 空行分隔，以下是 Markdown body（流程模板）
    """
    try:  # 捕获所有异常，确保单个文件解析失败不影响其他 skill 的加载
        raw = file_path.read_text(encoding="utf-8")  # 读取 SKILL.md 的完整文本内容
        result = parse_frontmatter(raw)  # 用 frontmatter 解析器分离元数据（meta）和正文（body）
        meta = result.meta  # 获取解析出的 YAML frontmatter 字典

        # ---- 组装并返回 SkillDefinition 实例 ----
        return SkillDefinition(
            name = meta.get("name") or file_path.parent.name or "unknown",  # skill 名称
            description=meta.get("description", ""),  # 描述文本，缺省为空字符串
            when_to_use=meta.get("when_to_use") or meta.get("when-to-use"),  # 兼容连字符和下划线两种写法
            prompt_template=result.body,  # SKILL.md 的 body 部分作为 prompt 模板
        )
    except Exception:  # 捕获任何未预期的异常（文件读取失败、解析错误等）
        return None  # 返回 None 表示解析失败，调用方会跳过此 skill
