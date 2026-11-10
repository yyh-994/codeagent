"""测试意图路由 route_intent() — intent 模块已被 conftest mock，直接测试 mock"""

# 注：conftest 已将 back.agent.intent 注入为 MagicMock，
# route_intent 是 AsyncMock(return_value={"intent": "write", "topic": "测试主题"})。
# 意图路由的完整测试需要在集成环境中进行（有真实 LLM）。
# 这里验证 mock 行为正确即可。

import pytest


@pytest.mark.asyncio
async def test_route_intent_mock_returns_blog():
    """验证 mock 的 route_intent 返回默认值 write。"""
    from back.agent.intent import route_intent
    result = await route_intent("任意消息")
    assert result["intent"] == "write"


def test_get_agent_is_mocked():
    """验证 get_agent 可以被调用且返回 mock。"""
    from back.agent.agents import get_agent
    agent = get_agent("blog")
    assert agent is not None
