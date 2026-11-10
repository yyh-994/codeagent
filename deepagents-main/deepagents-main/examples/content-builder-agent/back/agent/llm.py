"""共享 LLM 客户端 — 懒初始化单例，路由/记忆提取等轻量任务复用。"""

from langchain.chat_models import init_chat_model

from back.common.config import LLM_MODEL

_router_llm = None


def get_router():
    """返回进程级共享的裸 LLM 客户端（一问一答，无工具/历史/中间件）。"""
    global _router_llm
    if _router_llm is None:
        _router_llm = init_chat_model(LLM_MODEL)
    return _router_llm
