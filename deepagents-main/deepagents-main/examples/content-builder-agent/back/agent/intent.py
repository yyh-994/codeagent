"""意图识别与可选槽位提取；无意图且无槽位时才由上层追问。"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from back.agent.llm import get_router
from back.common.exceptions import ServiceException


# ═══════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════

class Slots(BaseModel):
    """可选条件：缺失时留空，不为了满足 schema 编造信息。"""

    topic: str | None = Field(None, description="write: 写作主题")
    time_scope: Literal[
        "today", "tomorrow", "this_week", "next_week", "date", "week", "term",
    ] | None = Field(None, description="query: 今天/明天/本周/下周/指定日期/指定教学周/整学期；未提及时留空")
    query_date: date | None = Field(None, description="query: 用户明确给出的日期 YYYY-MM-DD")
    week_no: int | None = Field(None, ge=1, description="query: 第几教学周")
    weekday: int | None = Field(None, ge=1, le=7, description="query: 周一=1，周日=7")
    course_name: str | None = Field(None, description="query: 课程名称")


class IntentSlots(Slots):
    """auto 模式：独立识别意图和槽位，意图不明时仍保留槽位。"""

    intent: Literal["write", "rag", "query", "unknown"] = Field(
        default="unknown",
        description="用户意图；不明确则用 unknown，仍提取能够确定的槽位",
    )


# ═══════════════════════════════════════════════════════
# Prompt
# ═══════════════════════════════════════════════════════

# ⚠️ 用字符串拼接，不要 .format()——防模板花括号 KeyError
_INTENT_PROMPT = """你是意图分类器。根据对话判断用户当前消息的意图并提取槽位。

意图定义:
- write: 写博客/文章/教程/社交媒体帖子
- rag: 根据PDF/文档提问、查资料
- query: 查询我自己的课表、上课日期、教学周、星期、课程、节次和教室
- unknown: 无法明确归入以上三类，不要猜；仍然提取能确定的槽位

判断规则:
- 当前消息是承接上文的（如"再来一篇"、"换个主题"），意图通常延续上轮意图，槽位从最近对话原文提取
- 当前消息话题明显切换时，忽略上轮意图，按当前消息判断
- 只提取用户明确提供的信息，缺失的槽位留空
- query 不再提取学生姓名或学号，当前只有一个学期，不需要提取学期
- 明天用 time_scope=tomorrow，本周/下周用 this_week/next_week；日期和教学周由后端计算
- 第三周周三用 time_scope=week、week_no=3、weekday=3；指定日期用 date 和 query_date
- 仅有星期时可继承最近明确的教学周；仅有课程名时仍保留 course_name
- 用户没有提供时间时，time_scope 留空，不要自行填 today
- 意图和槽位独立提取，不要因为 intent=unknown 就丢弃已识别的槽位

"""

_SLOT_ONLY_PROMPT = """你是槽位提取器。用户意图已确定，你只需从当前消息中提取对应槽位。

槽位说明:
- write → 提取 topic（写作主题）
- query → 提取 time_scope、query_date、week_no、weekday、course_name（个人课表条件）
- rag  → 无需提取

规则:
- 只提取用户明确提供的信息，缺失的槽位留空
- 承接上文时（如"再来一篇"），从最近对话原文推断
- 明天/本周/下周分别用 tomorrow/this_week/next_week，交给后端计算日期
- 当前只有一个学期，不提取姓名、学号或学期；未提及的时间条件留空

"""


# ═══════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════

_SLOT_KEYS: dict[str, list[str]] = {
    "write": ["topic"],
    "rag":   [],
    "query": ["time_scope", "query_date", "week_no", "weekday", "course_name"],
    "unknown": [],
}

_ROUTABLE_INTENTS = ("write", "rag", "query")
QUERY_SLOT_KEYS = tuple(_SLOT_KEYS["query"])
ALL_SLOT_KEYS = ("topic", *QUERY_SLOT_KEYS)


# ═══════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════

async def _fetch_history(thread_id: str) -> list:
    """从 checkpointer 读取最近消息历史。失败返回空列表（不阻断请求）。"""
    try:
        from back.agent.agents import get_checkpointer
        cp = get_checkpointer()
        state = await cp.aget_tuple({"configurable": {"thread_id": thread_id}})
        if state and state.checkpoint:
            return state.checkpoint.get("channel_values", {}).get("messages", [])
    except Exception:
        pass
    return []


# ═══════════════════════════════════════════════════════
# 对外接口
# ═══════════════════════════════════════════════════════

async def route_intent(
    user_message: str, *,
    thread_id: str,
    last_intent: Literal["write", "rag", "query"] | None = None,
    now_intent: str = "auto",
    fixed_intent: str | None = None,
) -> dict:
    """根据用户消息判断意图并提取槽位。

    now_intent 保留兼容；fixed_intent 是 Service 调用使用的别名。
    auto 时分类并提取槽位，已知意图时仅提取槽位。

    Returns:
        {"intent": "write", "topic": "..."}  或  {"intent": "unknown"}
    """
    if fixed_intent is not None:
        if now_intent != "auto" and now_intent != fixed_intent:
            raise ValueError("now_intent 与 fixed_intent 不一致")
        now_intent = fixed_intent
    if now_intent not in ("auto", *_ROUTABLE_INTENTS):
        raise ValueError("不支持的意图")
    history_msgs = await _fetch_history(thread_id)

    # ── 组装上下文：最近 2 条历史 + 当前消息 ──
    lines = []
    if now_intent == "auto":
        if last_intent in _ROUTABLE_INTENTS:
            lines.append(f"上轮意图: {last_intent}")
    else:
        lines.append(f"当前意图: {now_intent}")

    recent = [
        m for m in (history_msgs or [])
        if getattr(m, "type", "") in ("human", "ai") and str(getattr(m, "content", "")).strip()
    ][-2:]
    if recent:
        lines.append("## 最近对话:")
        lines.extend(
            f"{'用户' if m.type == 'human' else 'AI'}: {str(m.content)[:200]}"
            for m in recent
        )
    lines.append(f"\n当前消息: {user_message}")
    context = "\n".join(lines)

    # ── auto 模式：分类意图 + 槽位 ──
    if now_intent == "auto":
        llm = get_router().with_structured_output(IntentSlots)
        try:
            result = await llm.ainvoke(_INTENT_PROMPT + context)
        except Exception as error:
            raise ServiceException("意图识别服务异常，请稍后重试。", http_status=503) from error

        if result is None:
            raise ServiceException("意图识别未返回有效结果，请重试。", http_status=503)

        intent_slots = {"intent": result.intent}
        slots = ALL_SLOT_KEYS if result.intent == "unknown" else _SLOT_KEYS[result.intent]
        values = result.model_dump(mode="json")
        for slot in slots:
            intent_slots[slot] = values.get(slot)
        return intent_slots

    # ── 意图已知模式：只填槽位 ──
    llm = get_router().with_structured_output(Slots)
    try:
        result = await llm.ainvoke(_SLOT_ONLY_PROMPT + context)
    except Exception as error:
        raise ServiceException("槽位提取服务异常，请稍后重试。", http_status=503) from error

    if result is None:
        raise ServiceException("槽位提取未返回有效结果，请重试。", http_status=503)

    intent_slots = {"intent": now_intent}
    for slot in _SLOT_KEYS.get(now_intent, []):
        intent_slots[slot] = result.model_dump(mode="json").get(slot)
    return intent_slots
