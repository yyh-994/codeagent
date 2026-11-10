"""长期记忆写入侧 — 每轮从用户消息(HumanMessage)提取事实，向量化写入 store。

提取与裁决均走 with_structured_output：schema 由 API 层(function calling)保证，
无需手动解析 JSON/剥围栏。读取侧(语义检索 + 注入 system prompt)在 middleware.UserMemoryMiddleware。
"""

from typing import Literal

from pydantic import BaseModel, Field

from back.agent.llm import get_router
from back.common.config import MEMORY_NS


class MemoryExtraction(BaseModel):
    """提取结果：值得长期记住的用户事实列表。"""
    memories: list[str] = Field(
        default_factory=list,
        description="值得长期记住的用户信息，每条一句独立完整的陈述句；没有则为空列表",
    )


class MemoryOp(BaseModel):
    """对单条新信息的裁决操作。"""
    action: Literal["add", "replace", "skip"]
    text: str = Field("", description="add/replace 时的将要写入库的新记忆原文")
    delete_keys: list[str] = Field(default_factory=list, description="replace 时被取代的旧记忆 key")


class MemoryOps(BaseModel):
    """裁决结果：对每条新信息各一个操作。"""
    ops: list[MemoryOp] = Field(default_factory=list)


_MEMORY_EXTRACT_PROMPT = """从下面这条用户消息中提取值得长期记住的用户信息（身份、偏好、背景事实等）。

要求:
- 只提取长期有效的信息；一次性任务请求、寒暄、提问本身不算
- 每条记忆是一句独立完整的陈述句，如 "用户叫小明"、"用户偏好用 Python 写示例"
- 没有值得记住的信息时返回空列表

用户消息:
"""

_MEMORY_RECONCILE_PROMPT = """你在维护用户的长期记忆库。逐条判断"新信息"与"已有记忆"的关系，输出操作。

规则:
- 与某条旧记忆矛盾、或是旧记忆的更新 → replace：text 写新记忆，delete_keys 列出被取代的旧记忆 key
- 全新内容，与旧记忆不冲突 → add：delete_keys 留空
- 与旧记忆重复、没有新增内容 → skip

"""


async def extract_memories(uid: str, user_message: str) -> None:
    """每轮对话后从当前用户消息(HumanMessage)提取长期记忆，向量化写入 store。

    流程（mem0 模式：top-k 预筛 + LLM 裁决，k 是容量参数而非语义阈值）:
    1. LLM 结构化提取新事实，memories 为空则不入库；
    2. 每条新事实向量检索最相似的 K 条旧记忆，取并集作为裁决上下文；
    3. 并集为空（库空）→ 全部直接入库；否则送一次 LLM 裁决 add / replace / skip
       —— skip 兼去重，replace 删除被取代的旧记忆(遗忘/更新)；裁决失败降级为只增不删。
    """
    if uid == "anonymous" or not user_message.strip():
        return
    try:
        from back.agent.agents import get_store
        store = get_store()
    except Exception:
        return  # store 未初始化，静默跳过（与失败不阻断请求的约定一致）

    from datetime import datetime, timezone
    from uuid import uuid4

    async def _put(text: str) -> None:
        """写入一条记忆（store 表 + 向量表各一行）。"""
        await store.aput(
            (uid, MEMORY_NS), uuid4().hex,
            {"text": text, "created_at": datetime.now(timezone.utc).isoformat()},
        )

    # ── 1. 结构化提取：schema 保证返回 MemoryExtraction，无信息则 memories 为空 ──
    try:
        extraction = await get_router().with_structured_output(MemoryExtraction).ainvoke(
            _MEMORY_EXTRACT_PROMPT + user_message[:2000]
        )
    except Exception:
        extraction = None  # 网络/校验失败 → 本轮不入库
    facts = [f.strip() for f in extraction.memories if f.strip()] if extraction else []
    if not facts:
        return  # 无信息可存，不入向量库

    # ── 2. top-k 预筛：每条新事实检索最相似的 K 条旧记忆，并集作为裁决上下文 ──
    from back.common.config import MEMORY_RECONCILE_TOP_K

    candidates: dict[str, str] = {}   # {旧key: 旧文本}，dict 天然对并集去重
    for fact in facts:
        try:
            hits = await store.asearch((uid, MEMORY_NS), query=fact, limit=MEMORY_RECONCILE_TOP_K)
        except Exception:
            hits = []
        for h in hits:
            candidates[h.key] = h.value.get("text", "")

    try:
        if not candidates:
            # 库是空的 → 全部直接入库，不花裁决调用
            for fact in facts:
                await _put(fact)
            return

        # ── 3. LLM 结构化裁决增/删/改（新事实 + 预筛出的相似旧记忆）──
        lines = ["新信息:"]
        lines.extend(f"{i}. {fact}" for i, fact in enumerate(facts, 1))
        lines.append("\n已有记忆(与新信息最相似的若干条):")
        lines.extend(f"- key={k}: {t}" for k, t in candidates.items())
        try:
            result = await get_router().with_structured_output(MemoryOps).ainvoke(
                _MEMORY_RECONCILE_PROMPT + "\n".join(lines)
            )
        except Exception:
            result = None

        if result is None or not result.ops:
            # 裁决失败 → 降级：只增不删
            for fact in facts:
                await _put(fact)
            return

        for op in result.ops:
            text = op.text.strip()
            if op.action == "replace":
                for key in op.delete_keys:
                    if key in candidates:              # 只删真实展示过的 key，防 LLM 编造
                        await store.adelete((uid, MEMORY_NS), key)
            if op.action in ("add", "replace") and text:
                await _put(text)
    except Exception:
        pass
