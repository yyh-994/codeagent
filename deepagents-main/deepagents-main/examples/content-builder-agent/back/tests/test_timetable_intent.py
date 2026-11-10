"""使用真实意图路由与 Service，替换 LLM 和 Redis 等外部依赖。"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from back.common.exceptions import ServiceException
from back.service.intent import intent_service


@pytest.fixture
def routing(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """绕过全局 conftest 对 intent 的整体替身，加载真实路由代码。"""
    path = Path(__file__).parents[1] / "agent" / "intent.py"
    name = "back.agent._timetable_intent_tests"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "back.agent.intent", module)
    chain = SimpleNamespace(ainvoke=AsyncMock())
    router = SimpleNamespace(with_structured_output=Mock(return_value=chain))
    monkeypatch.setattr(module, "get_router", lambda: router)
    history = AsyncMock(return_value=[])
    monkeypatch.setattr(module, "_fetch_history", history)
    agents = ModuleType("back.agent.agents")
    agents.get_agent = Mock(return_value=object())
    monkeypatch.setitem(sys.modules, "back.agent.agents", agents)
    monkeypatch.setattr(intent_service, "_get_last_intent", AsyncMock(return_value=None))
    monkeypatch.setattr(intent_service, "_save_last_intent", AsyncMock())
    from back.agent import school_mcp
    from back.service.school import timetable_service
    monkeypatch.setattr(school_mcp, "get_query_tools", AsyncMock(return_value=[]))
    monkeypatch.setattr(timetable_service, "get_sql_context", AsyncMock(return_value={
        "term_id": "2025秋季测试", "today": "2025-09-17",
        "current_week_no": 3, "has_term_overlap": True,
    }))
    return SimpleNamespace(module=module, chain=chain, router=router,
                           agents=agents, history=history)


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, None),
        ({"intent": "unknown", "course_name": "  "}, None),
        ({"intent": "query"}, "query"),
        ({"intent": "unknown", "week_no": 3, "weekday": 4}, "query"),
        ({"intent": "unknown", "course_name": "高级工程数学"}, "query"),
        ({"intent": "unknown", "time_scope": "tomorrow"}, "query"),
        ({"intent": "unknown", "topic": "AI Agent"}, "write"),
        ({"intent": "write"}, "write"),
        ({"intent": "rag"}, "rag"),
    ],
)
async def test_clarify_only_when_both_intent_and_slots_are_missing(
    routing: SimpleNamespace, fields: dict, expected: str | None,
) -> None:
    """缺槽位不再阻止明确意图；未知意图也不会丢失有效槽位。"""
    routing.chain.ainvoke.return_value = routing.module.IntentSlots(**fields)
    reply = await intent_service.process_task("当前用户消息", thread_id="test")
    if expected is None:
        assert isinstance(reply, intent_service.QuickReply)
        routing.agents.get_agent.assert_not_called()
        intent_service._save_last_intent.assert_not_awaited()
    else:
        assert isinstance(reply, intent_service.TaskReply)
        assert reply.intent == expected
        routing.agents.get_agent.assert_called_once_with(expected)
        intent_service._save_last_intent.assert_awaited_once_with("test", expected)
        if expected == "query":
            assert "query_school_db" in reply.messages[0].content
            assert "课表时间上下文" in reply.messages[0].content
            assert "学号" not in reply.messages[0].content


async def test_unknown_intent_preserves_date_and_week_slots(routing: SimpleNamespace) -> None:
    """unknown 下仍返回条件，并将日期序列化为工具可使用的字符串。"""
    routing.chain.ainvoke.return_value = routing.module.IntentSlots(
        intent="unknown", query_date="2025-09-17", weekday=3,
    )
    result = await routing.module.route_intent("9月17日", thread_id="test")
    assert result["intent"] == "unknown"
    assert result["query_date"] == "2025-09-17"
    assert result["weekday"] == 3


async def test_fixed_intent_alias_allows_empty_slots(routing: SimpleNamespace) -> None:
    """Service 的 fixed_intent 参数与旧 now_intent 均可使用。"""
    routing.chain.ainvoke.return_value = routing.module.Slots()
    for argument in ({"fixed_intent": "query"}, {"now_intent": "query"}):
        result = await routing.module.route_intent("查课表", thread_id="test", **argument)
        assert result["intent"] == "query"
        assert all(result[key] is None for key in routing.module.QUERY_SLOT_KEYS)


async def test_followup_prompt_contains_history_and_last_intent(routing: SimpleNamespace) -> None:
    """连续追问仍将历史与上轮意图提供给提取模型。"""
    routing.history.return_value = [HumanMessage(content="第三周周三有什么课？"),
                                    AIMessage(content="高级工程数学。")]
    routing.chain.ainvoke.return_value = routing.module.IntentSlots(
        intent="query", week_no=3, weekday=4,
    )
    result = await routing.module.route_intent("那周四呢？", thread_id="test", last_intent="query")
    prompt = routing.chain.ainvoke.await_args.args[0]
    assert "上轮意图: query" in prompt
    assert "第三周周三" in prompt
    assert "那周四呢" in prompt
    assert result["week_no"] == 3
    assert result["weekday"] == 4


async def test_model_failure_is_not_treated_as_missing_slots(routing: SimpleNamespace) -> None:
    """模型服务故障不会被降级成要求用户补充信息。"""
    routing.chain.ainvoke.side_effect = RuntimeError("unavailable")
    with pytest.raises(ServiceException, match="意图识别服务异常"):
        await intent_service.process_task("查课表", thread_id="test")


def test_slot_schema_rejects_invalid_weekday(routing: SimpleNamespace) -> None:
    """无效星期不能进入已提取的查询条件。"""
    with pytest.raises(ValidationError):
        routing.module.IntentSlots(intent="query", weekday=8)
