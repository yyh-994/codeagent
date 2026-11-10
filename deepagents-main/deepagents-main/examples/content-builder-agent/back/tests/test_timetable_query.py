"""真实课表工具与查询逻辑测试，数据库使用内存 SQLite。"""

from collections.abc import AsyncIterator
from datetime import date

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from back.common.exceptions import ServiceException
from back.repository.seed_timetable import seed_timetable
from back.service.school import timetable_service


@pytest.fixture
async def database(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncEngine]:
    """使用真实数据库会话运行工具，同时固定当前日期为测试学期第三周。"""
    from back.repository import db_client

    engine = create_async_engine("sqlite+aiosqlite://")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(seed_timetable)
        monkeypatch.setattr(db_client, "AsyncSessionLocal", async_sessionmaker(engine))
        monkeypatch.setattr(timetable_service, "china_today", lambda: date(2025, 9, 17))
        yield engine
    finally:
        await engine.dispose()


async def test_tool_defaults_to_today_without_slots(database: AsyncEngine) -> None:
    """空参数可以直接查询今天，无需姓名或时间追问。"""
    from back.agent.tools import query_my_schedule

    result = await query_my_schedule.ainvoke({})
    assert result["current_week_no"] == 3
    assert result["query_start"] == "2025-09-17"
    assert result["courses"][0]["course_name"] == "高级工程数学"
    assert result["courses"][0]["start_time"] == "08:00"
    assert result["courses"][0]["end_time"] == "09:40"


@pytest.mark.parametrize(
    ("conditions", "start", "count"),
    [
        ({"week_no": 3}, "2025-09-15", 5),
        ({"week_no": 3, "weekday": 4}, "2025-09-15", 1),
        ({"weekday": 4}, "2025-09-15", 1),
        ({"time_scope": "tomorrow"}, "2025-09-18", 1),
        ({"time_scope": "next_week"}, "2025-09-22", 5),
        ({"time_scope": "term", "course_name": "工程数学"}, "2025-09-01", 5),
        ({"time_scope": "term", "course_name": "%"}, "2025-09-01", 0),
    ],
)
async def test_queries_resolve_time_and_filter_courses(
    database: AsyncEngine, conditions: dict, start: str, count: int,
) -> None:
    """相对日期、教学周、星期和课程筛选由后端完成。"""
    result = await timetable_service.query_schedule(**conditions)
    assert result["query_start"] == start
    assert len(result["courses"]) == count
    if conditions.get("weekday") == 4:
        assert result["courses"][0]["course_name"] == "算法设计与分析"


async def test_out_of_term_differs_from_no_classes(database: AsyncEngine) -> None:
    """超出学期不能被解释为当天没有课。"""
    outside = await timetable_service.query_schedule(query_date="2026-11-06")
    weekend = await timetable_service.query_schedule(query_date="2025-09-20")
    assert outside["in_term"] is False
    assert "不在已录入学期" in outside["message"]
    assert weekend["in_term"] is True
    assert "没有匹配的排课" in weekend["message"]
    assert outside["courses"] == weekend["courses"] == []


def test_invalid_query_date_reports_validation_error() -> None:
    """有槽位但格式无效是校验错误，不是意图缺失。"""
    with pytest.raises(ServiceException, match="日期须为"):
        timetable_service.resolve_window(
            date(2025, 9, 1), date(2025, 10, 5), query_date="not-a-date",
        )
