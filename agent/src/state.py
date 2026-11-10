"""Agent state definition for the LangGraph agent.

图状态：每次 chat() 调用时从 agent 实例属性复制进 initial_state，
节点通过 return dict 更新，LangGraph 自动合并。
与 agent 属性重复的字段标注了「← agent.xxx」。
"""

from typing import Annotated, Any
from typing import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentStateDict(TypedDict, total=False):
    """LangGraph 图的完整状态定义。

    分为两类：
      - 与 agent 实例属性重复：每次 chat() 从 agent 复制进图
      - LangGraph 独有：仅在图内存在（如 messages）
    """

    # ═══════════════════════════════════════════════════════════
    # LangGraph 独有
    # ═══════════════════════════════════════════════════════════

    # ── 消息流 ──
    messages: Annotated[list[BaseMessage], add_messages]
    """对话消息列表。LangGraph 的 add_messages reducer 自动拼接增量消息。"""

    # ═══════════════════════════════════════════════════════════
    # 以下全部与 CodingAgent 实例属性重复（← agent.xxx）
    # 每次 chat() 启动时从 agent 复制进图，节点只能通过 state 访问
    # ═══════════════════════════════════════════════════════════

    # ── 权限 ──
    permission_mode: str
    """权限模式: default / bypassPermissions / plan ← agent.permission_mode"""
    plan_file_path: str | None
    """Plan 模式下的计划文件路径 ← agent.plan_file_path"""

    # ── 会话标识 ──
    session_id: str
    """会话 ID，8 位随机 hex ← agent.session_id"""
    session_start_time: str
    """会话开始时间 ISO 格式 ← agent.session_start_time"""

    # ── 子 agent 输出 ──
    is_sub_agent: bool
    """是否为子 agent，影响流式输出和资源初始化 ← agent.is_sub_agent"""
    output_buffer: list[str] | None
    """子 agent 流式输出收集器，主 agent 为 None ← agent._output_buffer"""

    # ── 文件追踪 ──
    read_file_state: dict[str, float]
    """已读文件路径 → mtime，用于编辑前的冲突检测 ← agent._read_file_state"""

    # ── 预算控制 ──
    max_cost_usd: float | None
    """费用上限（美元），超限停止 ← agent.max_cost_usd"""
    max_turns: int | None
    """最大工具调用轮数 ← agent.max_turns"""
    current_turns: int
    """当前已执行轮数 ← agent.current_turns"""

    # ── Token 统计 ──
    total_input_tokens: int
    """累计输入 token 数 ← agent.total_input_tokens"""
    total_output_tokens: int
    """累计输出 token 数 ← agent.total_output_tokens"""
    total_cache_read_tokens: int
    """Anthropic 缓存读取 token ← agent.total_cache_read_tokens"""
    total_cache_creation_tokens: int
    """Anthropic 缓存写入 token ← agent.total_cache_creation_tokens"""
    last_llm_call_tokens: int
    """上一轮 LLM 调用的 input+output token 数，用于判断是否触发压缩 ← agent.last_llm_call_tokens"""
    last_llm_call_time: float
    """上次 LLM 调用时间戳，用于压缩间隔判断 ← agent.last_llm_call_time"""






