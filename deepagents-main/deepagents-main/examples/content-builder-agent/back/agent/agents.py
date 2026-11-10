"""Agent 工厂 + 共享存储 — 按意图拆分 Agent 实例：write / rag / query"""

import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from back.common.config import LLM_MODEL
from back.agent.tools import (
    ROOT_DIR,
    generate_image,
    web_search,
    search_knowledge_base,
    upload_and_index_pdf,
    load_subagents,
)
from back.agent.middleware import VIPPermissionMiddleware, UserMemoryMiddleware
from back.agent.query_prompt import QUERY_SYSTEM_PROMPT
from back.agent.school_mcp import loaded_query_tools

# ══════════════════════════════════════════════════════════════
# 共享状态（模块级单例，lifespan 负责异步初始化）
# ══════════════════════════════════════════════════════════════

_store = None
_checkpointer = None
_subagents_cache = None
_agents_cache = {}

# Store / checkpointer 的 SQLite 文件目录
_memory_dir = os.path.join(ROOT_DIR, "back", "output", "memory")
os.makedirs(_memory_dir, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 异步初始化（由 server.py lifespan 调用）
# ══════════════════════════════════════════════════════════════

async def init_async_stores():
    """在事件循环中创建 SQLite store 和 checkpointer。

    Store 使用单文件 + namespace 隔离：所有用户共用一份 agent_store.db，
    读写时由调用方显式传 (uid, ...) namespace 实现逻辑隔离。

    Store 启用 sqlite-vec 向量索引：value 中含 ``text`` 字段的条目
    会自动嵌入（复用 RAG 的 bge 模型），支持 ``asearch(ns, query=...)`` 语义检索。
    """
    global _store, _checkpointer
    from langgraph.store.sqlite.aio import AsyncSqliteStore
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    import aiosqlite

    from back.common.config import MEMORY_EMBEDDING_DIMS
    from back.service.rag.embedder import embedder  # 复用 RAG 嵌入模型（模块级单例，全进程只加载一次）

    conn_store = await aiosqlite.connect(os.path.join(_memory_dir, "agent_store.db"))
    conn_check = await aiosqlite.connect(os.path.join(_memory_dir, "checkpoints.db"))
    _store = AsyncSqliteStore(
        conn_store,
        index={"dims": MEMORY_EMBEDDING_DIMS, "embed": embedder, "text_fields": ["text"]},
    )
    _checkpointer = AsyncSqliteSaver(conn_check)
    await _store.setup()
    await _checkpointer.setup()


# ══════════════════════════════════════════════════════════════
# Agent 工厂
# ══════════════════════════════════════════════════════════════

_INTENT_SKILL = {
    "write": "blog-post",
    "rag": "rag-qa",
    "query": "school-query",
}


def get_agent(intent: str):
    """获取指定意图的 Agent 实例（按 intent 缓存，所有用户共享）。"""
    if intent in _agents_cache:
        return _agents_cache[intent]

    if _store is None or _checkpointer is None:
        raise RuntimeError("Agent stores not initialized — call init_async_stores() first")

    global _subagents_cache
    if _subagents_cache is None:
        _subagents_cache = load_subagents(Path(__file__).parent / "subagents.yaml")

    # 工具集
    tools_map = {
        "write": [generate_image, web_search],
        "rag":   [search_knowledge_base, upload_and_index_pdf],
        "query": loaded_query_tools() if intent == "query" else [],
    }
    tools = tools_map.get(intent, tools_map["write"])

    # write 意图需要负责研究的子 Agent
    subagents = _subagents_cache if intent == "write" else []

    # 中间件：UserMemoryMiddleware 先注入记忆，VIPPermissionMiddleware 再过滤工具
    middleware = [
        UserMemoryMiddleware(store=get_store()),
        VIPPermissionMiddleware(),
    ]

    agent = create_deep_agent(
        model=LLM_MODEL,
        system_prompt=QUERY_SYSTEM_PROMPT if intent == "query" else None,
        backend=FilesystemBackend(root_dir=ROOT_DIR, virtual_mode=True),
        checkpointer=_checkpointer,
        memory=["./back/agent/AGENTS.md"],#.就是root_dir=ROOT_DIR
        tools=tools,
        skills=[] if intent == "query" else [f"./skills/{_INTENT_SKILL[intent]}"],
        subagents=subagents,
        middleware=middleware,
        interrupt_on={"write_file": True, "edit_file": True},
    )
    _agents_cache[intent] = agent
    return agent


# ══════════════════════════════════════════════════════════════
# 线程/状态工具
# ══════════════════════════════════════════════════════════════

def get_checkpointer():
    """返回共享的 AsyncSqliteSaver。未初始化时抛出 RuntimeError。"""
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized — call init_async_stores() first")
    return _checkpointer


def get_store():
    """返回共享的 AsyncSqliteStore（含向量索引）。未初始化时抛出 RuntimeError。"""
    if _store is None:
        raise RuntimeError("Store not initialized — call init_async_stores() first")
    return _store

async def get_thread_state(thread_id: str) -> dict | None:
    """直接从 checkpointer 读取指定会话的 state，不依赖 agent 实例。"""
    cp = get_checkpointer()
    tuple_ = await cp.aget_tuple({"configurable": {"thread_id": thread_id}})
    if not tuple_ or not tuple_.checkpoint:
        return None
    return tuple_.checkpoint.get("channel_values", {})


async def list_threads(limit: int = 200) -> list[dict]:
    """列出所有会话线程。"""
    cp = get_checkpointer()
    await cp.setup()
    async with cp.conn.execute(
        "SELECT DISTINCT thread_id FROM checkpoints ORDER BY checkpoint_id DESC LIMIT ?",
        (limit,),
    ) as cursor:
        rows = await cursor.fetchall()

    threads = []
    for (thread_id,) in rows:
        config = {"configurable": {"thread_id": thread_id}}
        tuple_ = await cp.aget_tuple(config)
        title = thread_id[:20]
        if tuple_ and tuple_.checkpoint:
            msgs = tuple_.checkpoint.get("channel_values", {}).get("messages", [])
            for m in msgs:
                if hasattr(m, "type") and m.type == "human":
                    c = m.content
                    if isinstance(c, list):
                        c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
                    title = str(c)[:40] + ("..." if len(str(c)) > 40 else "")
                    break
        threads.append({"thread_id": thread_id, "title": title})
    return threads

def serialize_messages_for_frontend(messages: list) -> list[dict]:
    """将 LangChain 消息列表转换为前端兼容的 {type, content} 格式。"""
    result = []
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)
        if not content or not str(content).strip():
            continue
        msg_type = type(msg).__name__
        if msg_type == "HumanMessage":
            result.append({"type": "human", "content": str(content)})
        elif msg_type == "AIMessage":
            result.append({"type": "ai", "content": str(content)})
    return result
