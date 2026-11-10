"""个人课表查询：后端确定日期和教学周，只读访问课表三张表。"""

from datetime import date, datetime, timedelta, timezone
from typing import Literal

from sqlalchemy import Select, and_, or_, select

from back.common.exceptions import ServiceException
from back.repository.entity import AcademicTerm, ClassPeriod, CourseSchedule


TimeScope = Literal["today", "tomorrow", "this_week", "next_week", "date", "week", "term"]


def china_today() -> date:
    """按北京时间获取日期，不依赖服务器所在时区。"""
    return datetime.now(timezone(timedelta(hours=8))).date()


def resolve_window(
    week1_monday: date, end_date: date, *,
    time_scope: TimeScope | None = None, query_date: str | None = None,
    week_no: int | None = None, weekday: int | None = None,
    today: date | None = None,
) -> tuple[date, date, str]:
    """将相对日期或教学周转换为查询起止日期，未给时间默认今天。"""
    today = today or china_today()
    if week1_monday.weekday() != 0:
        raise ServiceException("学期校历中的第一周日期必须是周一。")
    if weekday is not None and not 1 <= weekday <= 7:
        raise ServiceException("星期必须在 1～7 之间。")
    if week_no is not None and week_no < 1:
        raise ServiceException("教学周必须大于 0。")
    scope = time_scope or (
        "date" if query_date else "week" if week_no is not None
        else "this_week" if weekday is not None else "today"
    )
    if scope == "date":
        try:
            target = date.fromisoformat(query_date or "")
        except ValueError as error:
            raise ServiceException("指定日期须为 YYYY-MM-DD 格式。") from error
        return target, target, scope
    if scope in ("today", "tomorrow"):
        target = today + timedelta(days=scope == "tomorrow")
        return target, target, scope
    if scope in ("this_week", "next_week"):
        start = today - timedelta(days=today.weekday())
        start += timedelta(days=7 if scope == "next_week" else 0)
        return start, start + timedelta(days=6), scope
    if scope == "week":
        if week_no is None:
            raise ServiceException("指定教学周时需要有效的 week_no。")
        start = week1_monday + timedelta(weeks=week_no - 1)
        return start, start + timedelta(days=6), scope
    if scope == "term":
        return week1_monday, end_date, scope
    raise ServiceException("不支持的课表时间范围。")


def schedule_statement(
    term_id: str, week1_monday: date, start: date, end: date, *,
    weekday: int | None = None, course_name: str | None = None,
) -> Select:
    """按教学周与星期构建参数化查询，关联开始和结束节次时间。"""
    schedule = CourseSchedule.__table__
    first = ClassPeriod.__table__.alias("first_period")
    last = ClassPeriod.__table__.alias("last_period")
    first_week = (start - week1_monday).days // 7 + 1
    last_week = (end - week1_monday).days // 7 + 1
    statement = (
        select(
            schedule.c.course_name, schedule.c.week_no, schedule.c.weekday,
            schedule.c.start_period, schedule.c.end_period, schedule.c.classroom,
            first.c.start_time, last.c.end_time,
        )
        .select_from(
            schedule.join(first, schedule.c.start_period == first.c.period_no)
            .join(last, schedule.c.end_period == last.c.period_no)
        )
        .where(
            schedule.c.term_id == term_id,
            or_(schedule.c.week_no > first_week,
                and_(schedule.c.week_no == first_week,
                     schedule.c.weekday >= start.isoweekday())),
            or_(schedule.c.week_no < last_week,
                and_(schedule.c.week_no == last_week,
                     schedule.c.weekday <= end.isoweekday())),
        )
    )
    if weekday is not None:
        statement = statement.where(schedule.c.weekday == weekday)
    if course_name and course_name.strip():
        statement = statement.where(schedule.c.course_name.contains(course_name.strip(), autoescape=True))
    return statement.order_by(schedule.c.week_no, schedule.c.weekday, schedule.c.start_period)


async def query_schedule(
    *, time_scope: TimeScope | None = None, query_date: str | None = None,
    week_no: int | None = None, weekday: int | None = None,
    course_name: str | None = None,
) -> dict:
    """查询唯一学期的个人课表，超出学期返回范围提示而非编造周次。"""
    from back.repository.db_client import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        terms = (await session.execute(select(
            AcademicTerm.term_id, AcademicTerm.week1_monday, AcademicTerm.end_date,
        ).limit(2))).mappings().all()
        if len(terms) != 1:
            raise ServiceException("个人课表需要配置唯一一个学期。")
        term = terms[0]
        today = china_today()
        start, end, scope = resolve_window(
            term["week1_monday"], term["end_date"], time_scope=time_scope,
            query_date=query_date, week_no=week_no, weekday=weekday, today=today,
        )
        in_term = start >= term["week1_monday"] and end <= term["end_date"]
        response = {
            "term_id": term["term_id"], "today": today.isoformat(),
            "current_week_no": (
                (today - term["week1_monday"]).days // 7 + 1
                if term["week1_monday"] <= today <= term["end_date"] else None
            ),
            "time_scope": scope, "query_start": start.isoformat(),
            "query_end": end.isoformat(), "in_term": in_term, "courses": [],
        }
        clipped_start = max(start, term["week1_monday"])
        clipped_end = min(end, term["end_date"])
        if clipped_start > clipped_end:
            response["message"] = (
                f"查询日期不在已录入学期范围内（{term['week1_monday']}～{term['end_date']}）。"
                "可指定该学期内的日期或教学周。"
            )
            return response
        statement = schedule_statement(
            term["term_id"], term["week1_monday"], clipped_start, clipped_end,
            weekday=weekday, course_name=course_name,
        )
        rows = (await session.execute(statement)).mappings().all()
        response["courses"] = [_format_course(dict(row), term["week1_monday"]) for row in rows]
        response["message"] = "查询完成。" if rows else "该查询范围内没有匹配的排课。"
        return response


def _format_course(row: dict, week1_monday: date) -> dict:
    """将数据库日期时间转换为可供工具输出的 JSON 值。"""
    row["date"] = (week1_monday + timedelta(
        weeks=row["week_no"] - 1, days=row["weekday"] - 1,
    )).isoformat()
    row["start_time"] = row["start_time"].strftime("%H:%M")
    row["end_time"] = row["end_time"].strftime("%H:%M")
    return row


async def get_sql_context(
    *, time_scope: TimeScope | None = None, query_date: str | None = None,
    week_no: int | None = None, weekday: int | None = None,
    course_name: str | None = None,
) -> dict:
    """为 Text-to-SQL 提供确定的校历与日期范围，不替模型决定查询结构。"""
    from back.repository.db_client import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        terms = (await session.execute(select(
            AcademicTerm.term_id, AcademicTerm.week1_monday, AcademicTerm.end_date,
        ).limit(2))).mappings().all()
    if len(terms) != 1:
        raise ServiceException("个人课表需要配置唯一一个学期。")
    term = terms[0]
    today = china_today()
    start, end, scope = resolve_window(
        term["week1_monday"], term["end_date"], time_scope=time_scope,
        query_date=query_date, week_no=week_no, weekday=weekday, today=today,
    )
    first = max(start, term["week1_monday"])
    last = min(end, term["end_date"])
    available = first <= last
    return {
        "term_id": term["term_id"], "week1_monday": term["week1_monday"].isoformat(),
        "end_date": term["end_date"].isoformat(), "today": today.isoformat(),
        "current_week_no": ((today - term["week1_monday"]).days // 7 + 1
                            if term["week1_monday"] <= today <= term["end_date"] else None),
        "time_scope": scope, "query_start": start.isoformat(), "query_end": end.isoformat(),
        "has_term_overlap": available,
        "first_week": (first - term["week1_monday"]).days // 7 + 1 if available else None,
        "last_week": (last - term["week1_monday"]).days // 7 + 1 if available else None,
        "start_weekday": first.isoweekday() if available else None,
        "end_weekday": last.isoweekday() if available else None,
        "weekday_filter": weekday, "course_name_filter": course_name,
    }
