"""意图编排 Service — 把「路由 → 槽位检查 → prompt 组装 → Agent 获取」整条链从 Route 抽离。

Route 只做 HTTP 的事（解请求、调 Service、包 SSE），Service 返回纯数据：
- QuickReply: 追问文本，Route 包成 _sse_once 返回
- TaskReady:  编排完成，Route 直接开始 SSE 流
"""

from dataclasses import dataclass
import json
from typing import Any

from langchain_core.messages import HumanMessage


# ══════════════════════════════════════════════════════════════
# 返回值类型
# ══════════════════════════════════════════════════════════════

@dataclass
class QuickReply:
    """意图与槽位均未识别时返回追问文本，由 Route 包成 SSE。"""
    text: str


@dataclass
class TaskReply:
    """编排完成，Route 可直接拿这些字段开始 agent.astream SSE 流。"""
    agent: Any
    messages: list          # [HumanMessage, ...] 已注入槽位 prefix
    intent: str             # "write" | "rag" | "query"
    raw_user_msg: str       # 原始用户消息（语义缓存 key 用）


# ══════════════════════════════════════════════════════════════
# Redis 缓存（上轮意图，供下一轮承接判断）
# ══════════════════════════════════════════════════════════════

_INTENT_TTL = 1800  # 30 分钟


async def _get_last_intent(thread_id: str) -> str | None:
    """读上轮意图（Redis 不可用时静默返回 None，退化为无此信号）。"""
    try:
        from back.repository.redis_client import client
        if client is not None:
            v = await client.get(f"intent:{thread_id}")
            return v.decode() if isinstance(v, bytes) else v
    except Exception:
        return None
    return None


async def _save_last_intent(thread_id: str, intent: str) -> None:
    """存本轮意图，供下一轮承接判断（失败不阻断请求）。"""
    try:
        from back.repository.redis_client import client
        if client is not None:
            await client.setex(f"intent:{thread_id}", _INTENT_TTL, intent)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# 编排入口
# ══════════════════════════════════════════════════════════════

async def process_task(
    user_message: str, *,
    thread_id: str,
    intent_override: str = "auto",
) -> QuickReply | TaskReply:
    """编排意图识别全流程，返回 QuickReply（追问）或 TaskReply（就绪）。

    流程:
    1. route_intent(fixed_intent=...) → auto 走 LLM 分类，已知只填槽位
    2. 意图与有效槽位都没有 → QuickReply；仅有槽位时确定对应路由
    3. 槽位注入 prompt → TaskReply（含 agent 实例）
    """
    from back.agent.intent import QUERY_SLOT_KEYS, route_intent
    from back.agent.agents import get_agent

    raw_user_msg = user_message

    # ── 1. 意图路由（auto → LLM 分类+槽位；已知 → LLM 只填槽位）──
    intent_slots = await route_intent(
        user_message,
        thread_id=thread_id,
        last_intent=await _get_last_intent(thread_id),
        fixed_intent=intent_override,
    )
    intent = intent_slots.get("intent", "unknown")

    # ── 2. 没有明确意图但有槽位：仍进入对应 Agent ──
    if intent not in ("write", "rag", "query"):
        if any(_has_value(intent_slots.get(key)) for key in QUERY_SLOT_KEYS):
            intent = "query"
        elif _has_value(intent_slots.get("topic")):
            intent = "write"
        else:
            return QuickReply(
                "我还没识别到明确意图或查询条件。你想查课表、查资料，还是写内容？"
            )

    # ── 3. 存本轮意图（供下一轮承接判断）──
    await _save_last_intent(thread_id, intent)

    # ── 4. 槽位不再必填；缺少时间条件由课表工具应用默认值 ──
    msgs = [HumanMessage(content=user_message)]

    if intent == "write":
        topic = intent_slots.get("topic")
        prefix = "[意图: 写作]\n"
        if _has_value(topic):
            prefix += f"[主题: {topic}]\n"
        msgs[-1] = HumanMessage(content=prefix + "\n" + user_message)
    elif intent == "query":
        conditions = {
            key: intent_slots[key] for key in QUERY_SLOT_KEYS
            if _has_value(intent_slots.get(key))
        }
        lines = [
            "[意图: 查询我的课表]",
            "[课表查询参数: " + json.dumps(conditions, ensure_ascii=False) + "]",
            "生成 SQL 模板与 params，调用 FastMCP 的 query_school_db 执行；无条件默认今天。",
        ]
        from back.agent.school_mcp import get_query_tools
        from back.service.school.timetable_service import get_sql_context

        context = await get_sql_context(**conditions)
        lines.append("[课表时间上下文: " + json.dumps(context, ensure_ascii=False) + "]")
        await get_query_tools()
        msgs[-1] = HumanMessage(content="\n".join(lines) + "\n\n" + msgs[-1].content)

    # ── 6. 获取 Agent ──
    agent = get_agent(intent)

    return TaskReply(
        agent=agent,
        messages=msgs,
        intent=intent,
        raw_user_msg=raw_user_msg,
    )


def _has_value(value: object) -> bool:
    """None、空字符串、空白字符串均不算提取到了槽位。"""
    return value is not None and (not isinstance(value, str) or bool(value.strip()))
