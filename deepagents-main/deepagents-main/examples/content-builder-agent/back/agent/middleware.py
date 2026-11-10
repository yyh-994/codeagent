"""Agent 中间件 — VIP 权限 + 用户长期记忆注入"""

from langchain.agents.middleware.types import AgentMiddleware
from langgraph.store.base import BaseStore
from deepagents.middleware._utils import append_to_system_message

from back.common.config import MEMORY_NS


def _user_role(request) -> str:
    return request.config.get("configurable", {}).get("user_role", "basic")


def _user_id(request) -> str:
    return request.config.get("configurable", {}).get("user_id", "")

_BLOCKED_TOOLS = frozenset({"generate_image"})


def _tool_name(tool: object) -> str | None:
    """提取工具名，兼容 ``BaseTool`` 和 ``dict``。"""
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


class VIPPermissionMiddleware(AgentMiddleware):
    """非 VIP 用户自动屏蔽 ``generate_image``。"""

    async def awrap_model_call(self, request, handler):
        if _user_role(request) != "vip":
            request = request.override(
                tools=[t for t in request.tools if _tool_name(t) not in _BLOCKED_TOOLS]
            )
        return await handler(request)


def _last_human_text(messages: list) -> str:
    """倒序找最后一条 HumanMessage 并取纯文本（content 为 list 时拼接 text block）。"""
    for m in reversed(messages or []):
        if getattr(m, "type", "") != "human":
            continue
        content = getattr(m, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
    return ""


class UserMemoryMiddleware(AgentMiddleware):
    """每次请求前用当前用户消息对长期记忆做语义检索，top-k 注入 system prompt。

    记忆由 ``extract_memories`` 每轮从 HumanMessage 提取后向量化写入：
    namespace=(uid, "memories"), value={"text": 一句陈述句}。
    """

    def __init__(self, store: BaseStore | None = None) -> None:
        self._store = store

    async def awrap_model_call(self, request, handler):
        uid = _user_id(request)
        if not uid or self._store is None:
            return await handler(request)

        query = _last_human_text(request.messages).strip()
        if not query:
            return await handler(request)

        from back.common.config import MEMORY_TOP_K, MEMORY_MIN_SCORE

        try:
            items = await self._store.asearch((uid, MEMORY_NS), query=query[:1000], limit=MEMORY_TOP_K)
        except Exception:
            return await handler(request)

        lines = []
        for item in items:
            if item.score is not None and item.score < MEMORY_MIN_SCORE:
                continue  # 过滤低相关记忆
            text = item.value.get("text")
            if text:
                lines.append(f"- {text}")

        if not lines:
            return await handler(request)

        prompt = (
            "## 用户长期记忆\n\n"
            "以下是该用户的最新长期记忆（可能在其他会话中更新过）。"
            "若与本会话更早的对话内容冲突，以此处为准：\n\n" + "\n".join(lines)
        )
        new_system = append_to_system_message(request.system_message, prompt)
        return await handler(request.override(system_message=new_system))
