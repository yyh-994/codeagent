"""Text-to-SQL 的 AST 边界、参数绑定、执行与 MCP 协议测试。"""

from collections.abc import AsyncIterator
from datetime import date, time
import importlib.util
from pathlib import Path
import sys

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from back.common.exceptions import ServiceException
from back.repository.seed_timetable import seed_timetable
from back.service.school.sql_service import execute_school_sql, validate_school_sql


@pytest.fixture
async def database(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncEngine]:
    """用真实内存数据库执行模型可能生成的查询。"""
    from back.repository import db_client
    engine = create_async_engine("sqlite+aiosqlite://")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(seed_timetable)
        monkeypatch.setattr(db_client, "AsyncSessionLocal", async_sessionmaker(engine))
        yield engine
    finally:
        await engine.dispose()


@pytest.mark.parametrize("sql", [
    "DELETE FROM course_schedule",
    "SELECT * FROM class_periods; DELETE FROM course_schedule",
    "SELECT * FROM information_schema.tables WHERE table_name=:name",
    "SELECT * FROM other_db.course_schedule",
    "SELECT * FROM tables",
    "SELECT * FROM users AS course_schedule",
    "SELECT * FROM class_periods WHERE EXISTS (SELECT 1 FROM users)",
    "WITH class_periods AS (SELECT * FROM users) SELECT * FROM class_periods",
    "WITH unused AS (SELECT * FROM information_schema.tables) SELECT * FROM class_periods",
    "SELECT * FROM class_periods UNION SELECT * FROM users",
    "SELECT * FROM class_periods JOIN information_schema.tables ON 1=1",
    "SELECT SLEEP(1) FROM class_periods",
    "SELECT LOAD_FILE(:name) FROM class_periods",
    "SELECT @@version FROM class_periods",
    "SELECT @x:=period_no FROM class_periods",
    "SELECT * FROM class_periods FOR UPDATE",
    "SELECT period_no INTO @x FROM class_periods",
    "SELECT period_no INTO OUTFILE :name FROM class_periods",
    "SELECT /* course_schedule */ * FROM information_schema.tables",
    "SELECT /*!50000 SQL_NO_CACHE */ * FROM class_periods",
    "WITH RECURSIVE c AS (SELECT * FROM class_periods) SELECT * FROM c",
    "SELECT 1",
    "SELECT * FROM :name",
    "SELECT * FROM class_periods WHERE period_no=?",
    "SELECT * FROM class_periods LIMIT 10000",
    "SELECT * FROM class_periods LIMIT :name",
    "SELECT * FROM course_schedule WHERE course_name='inline value'",
])
def test_reject_unapproved_sources_or_syntax(sql: str) -> None:
    """子查询、CTE、别名、系统库、注释等不能绕过允许列表。"""
    with pytest.raises(ServiceException):
        validate_school_sql(sql, {"name": "course_schedule"} if ":name" in sql else {})


@pytest.mark.parametrize("sql", [
    "SELECT period_no FROM class_periods",
    "WITH c AS (SELECT period_no FROM class_periods) SELECT * FROM c",
    "SELECT * FROM (SELECT period_no FROM class_periods) AS c",
    "SELECT period_no FROM class_periods UNION SELECT period_no FROM class_periods",
    "SELECT COUNT(*) AS n FROM course_schedule",
])
def test_allow_real_business_sources_in_nested_queries(sql: str) -> None:
    """允许正常的单表、聚合、派生表、CTE、UNION，不误判虚拟表名。"""
    query = validate_school_sql(sql, {})
    assert query.row_limit == 100
    assert "LIMIT 101" in query.sql


@pytest.mark.parametrize("params", [
    {}, {"week": 3, "extra": 4}, {"week": [3]}, {"week": {"x": 3}}, {"week": float("inf")},
])
def test_parameter_contract(params: dict) -> None:
    """占位符与参数一一对应，并且只接受有界 JSON 标量。"""
    with pytest.raises(ServiceException):
        validate_school_sql("SELECT * FROM course_schedule WHERE week_no=:week", params)


async def test_execute_join_with_bound_values(database: AsyncEngine) -> None:
    """模型提供的 JOIN 与参数真正进入 SQLAlchemy 执行。"""
    sql = """SELECT s.course_name, s.classroom, p1.start_time, p2.end_time
FROM course_schedule s
JOIN class_periods p1 ON s.start_period=p1.period_no
JOIN class_periods p2 ON s.end_period=p2.period_no
WHERE s.week_no=:week AND s.weekday=:day"""
    result = await execute_school_sql(sql, {"week": 3, "day": 4})
    assert result["ok"] is True
    assert result["columns"] == ["course_name", "classroom", "start_time", "end_time"]
    row = result["rows"][0]
    assert row[:2] == ["算法设计与分析", "教学楼102"]
    assert time.fromisoformat(row[2]) == time(8)
    assert time.fromisoformat(row[3]) == time(9, 40)


async def test_bound_injection_string_is_only_a_value(database: AsyncEngine) -> None:
    """参数里的 SQL 片段不会扩大查询范围。"""
    result = await execute_school_sql(
        "SELECT course_name FROM course_schedule WHERE course_name=:course",
        {"course": "' OR 1=1 --"},
    )
    assert result["rows"] == []
    count = await execute_school_sql("SELECT COUNT(*) FROM course_schedule", {})
    assert count["rows"] == [[25]]


async def test_row_limit_reports_truncation(database: AsyncEngine) -> None:
    """实际探测多一行，返回上限内数据并标记截断。"""
    result = await execute_school_sql("SELECT course_name FROM course_schedule LIMIT 2", {})
    assert result["row_count"] == 2
    assert result["truncated"] is True


async def test_mcp_exposes_one_sql_tool_and_returns_results(database: AsyncEngine) -> None:
    """通过真实 MCP 内存协议调用 FastMCP，使用内存数据库而非网络服务。"""
    from mcp.shared.memory import create_connected_server_and_client_session
    from back.mcp.school_server import mcp

    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        tools = await session.list_tools()
        assert [tool.name for tool in tools.tools] == ["query_school_db"]
        result = await session.call_tool("query_school_db", {
            "sql": "SELECT period_no FROM class_periods WHERE period_no=:p", "params": {"p": 1},
        })
        assert not result.isError
        assert result.structuredContent["rows"] == [[1]]
        rejected = await session.call_tool("query_school_db", {
            "sql": "SELECT * FROM information_schema.tables", "params": {},
        })
        assert rejected.structuredContent["ok"] is False


async def test_date_context_is_provided_without_fixed_query(database: AsyncEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    """后端只提供换算后的时间范围，Agent 仍可自主选择查询表和 JOIN。"""
    from back.service.school import timetable_service
    monkeypatch.setattr(timetable_service, "china_today", lambda: date(2025, 9, 17))
    context = await timetable_service.get_sql_context(week_no=3, weekday=4)
    assert context["first_week"] == context["last_week"] == 3
    assert context["weekday_filter"] == 4
    assert context["term_id"] == "2025秋季测试"
    assert "sql" not in context


def test_query_factory_uses_mcp_and_prompt_without_skill(monkeypatch: pytest.MonkeyPatch) -> None:
    """检查真实工厂的 query 分支，避免测试被 conftest 的 agents 替身绕过。"""
    from types import SimpleNamespace
    from back.agent.query_prompt import QUERY_SYSTEM_PROMPT

    path = Path(__file__).parents[1] / "agent" / "agents.py"
    name = "back.agent._sql_factory_tests"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    tool = SimpleNamespace(name="query_school_db")
    monkeypatch.setattr(module, "_store", object())
    monkeypatch.setattr(module, "_checkpointer", object())
    monkeypatch.setattr(module, "_subagents_cache", [])
    monkeypatch.setattr(module, "loaded_query_tools", lambda: [tool])
    monkeypatch.setattr(module, "create_deep_agent", lambda **kwargs: kwargs)
    agent = module.get_agent("query")
    assert agent["tools"] == [tool]
    assert agent["system_prompt"] == QUERY_SYSTEM_PROMPT
    assert agent["skills"] == []
