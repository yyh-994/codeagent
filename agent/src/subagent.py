"""Sub-agent system — 只读探索子 agent。

对应 Claude Code 的 AgentTool 中的 explore 类型（只读搜索）。
子 agent 由 graph.py 的 agent 工具在运行时创建（fork-return 模式：跑完只回结果）。

与 skill 的区别：
  skill      → 固定"流程"（按需加载的 prompt 模板，body 是具体步骤）
  sub-agent  → 只读"搜索角色"（独立上下文里查海量文件，只回结论）

调用链：
  tools.py          → agent 工具定义（LLM 通过 agent tool 委派搜索任务）
  graph.py          → _execute_agent_tool() → 创建 CodingAgent 子实例执行
  subagent.py 内部  → get_sub_agent_config() → 返回 explore 的 system prompt + 工具集
"""

from __future__ import annotations

from .tools import tool_definitions  # 从工具定义列表中筛选子 agent 的工具集


# 只读工具集合（explore 使用）
READ_ONLY_TOOLS = frozenset({"read_file", "list_files", "grep_search"})

# ─── explore 的 system prompt ──────────────────────────────────

EXPLORE_PROMPT = """你是 Claude Code Agent 的只读探索子 Agent，负责快速搜索和阅读代码库。

=== 严格只读模式 ===
你只有 read_file、list_files、grep_search 三个工具，禁止任何写操作。

你的职责：
- 快速定位文件、搜索代码和模式
- 分析多个文件以理解系统架构
- 调查需要探索大量文件的复杂问题

行为准则：
- 先用 list_files 广撒网，再用 grep_search 精确定位
- 尽可能并行发起多个搜索/读取调用
- 高效完成任务，尽快返回结果，给出清晰结论"""


# ─── 主配置函数 ────────────────────────────────────────────────

def get_sub_agent_config() -> dict:
    """返回 explore 子 agent 配置：{"system_prompt": str, "tools": list[dict]}。

    Returns:
        包含 system_prompt（角色定位）和 tools（只读工具集）的字典。0
    """
    tools = [t for t in tool_definitions if t["name"] in READ_ONLY_TOOLS]
    return {"system_prompt": EXPLORE_PROMPT, "tools": tools}
