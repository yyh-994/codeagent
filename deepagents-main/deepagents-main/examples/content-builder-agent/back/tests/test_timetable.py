"""课表 schema 与重复导入测试：仅使用启用外键的内存 SQLite。"""

from collections.abc import Iterator

import pytest
from sqlalchemy import Connection, create_engine, func, inspect, insert, select, text
from sqlalchemy.exc import IntegrityError

from back.repository.entity import AcademicTerm, ClassPeriod, CourseSchedule
from back.repository.seed_timetable import TEST_TERM, seed_timetable


@pytest.fixture
def connection() -> Iterator[Connection]:
    """为每个测试创建独立的内存数据库。"""
    engine = create_engine("sqlite://")
    try:
        with engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys=ON"))
            conn.commit()
            with conn.begin():
                seed_timetable(conn)
                yield conn
    finally:
        engine.dispose()


def test_seed_is_repeatable_and_only_creates_timetable_tables(connection: Connection) -> None:
    """重复导入不新增副本，也不创建用户等无关表。"""
    assert set(inspect(connection).get_table_names()) == {
        "academic_terms", "class_periods", "course_schedule",
    }
    assert seed_timetable(connection) == {
        "academic_terms": 0, "class_periods": 0, "course_schedule": 0,
    }
    count = connection.scalar(select(func.count()).select_from(CourseSchedule))
    assert count == 25


def test_date_to_week_and_joined_class_times(connection: Connection) -> None:
    """周三课程可按真实日期换算周次并关联到连续两节的起止时间。"""
    from datetime import date, time

    target = date(2025, 9, 17)
    week = (target - TEST_TERM["week1_monday"]).days // 7 + 1
    schedule = CourseSchedule.__table__
    start = ClassPeriod.__table__.alias("start")
    end = ClassPeriod.__table__.alias("end")
    result = connection.execute(
        select(schedule.c.course_name, start.c.start_time, end.c.end_time)
        .select_from(schedule.join(start, schedule.c.start_period == start.c.period_no)
                     .join(end, schedule.c.end_period == end.c.period_no))
        .where(schedule.c.term_id == TEST_TERM["term_id"], schedule.c.week_no == week,
               schedule.c.weekday == target.isoweekday())
    ).one()
    assert week == 3
    assert tuple(result) == ("高级工程数学", time(8), time(9, 40))


def test_foreign_keys_reject_unknown_period_and_term(connection: Connection) -> None:
    """不存在的节次或学期不能进入排课表。"""
    row = {
        "term_id": TEST_TERM["term_id"], "course_name": "无效排课",
        "week_no": 1, "weekday": 6, "start_period": 1,
        "end_period": 99, "classroom": "测试教室",
    }
    for overrides in ({}, {"term_id": "不存在的学期", "end_period": 2}):
        with pytest.raises(IntegrityError), connection.begin_nested():
            connection.execute(insert(CourseSchedule).values(**(row | overrides)))


def test_seed_refuses_to_overwrite_existing_calendar(connection: Connection) -> None:
    """已经调整过的校历不会被测试导入脚本悄悄覆盖。"""
    from datetime import date

    connection.execute(
        AcademicTerm.__table__.update().where(AcademicTerm.term_id == TEST_TERM["term_id"])
        .values(end_date=date(2025, 10, 12))
    )
    with pytest.raises(ValueError, match="未覆盖"):
        seed_timetable(connection)
    assert connection.scalar(select(AcademicTerm.end_date)) == date(2025, 10, 12)
