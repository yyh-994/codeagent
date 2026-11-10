"""创建课表三张表并插入第 1～5 周的示例数据。

在项目目录执行：uv run python -m back.repository.seed_timetable
仅创建课表相关表；重复运行跳过相同记录，遇到不同内容则报错，不覆盖。
示例学期仅覆盖 2025-09-01 至 2025-10-05，不代表学校实际学期结束日期。
"""

import os
from datetime import date, time
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Connection, URL, create_engine, insert, select
from sqlalchemy.sql.schema import Table

from back.repository.entity import AcademicTerm, Base, ClassPeriod, CourseSchedule


TABLES = (
    AcademicTerm.__table__, ClassPeriod.__table__, CourseSchedule.__table__,
)
TEST_TERM = {
    "term_id": "2025秋季测试",
    "week1_monday": date(2025, 9, 1),
    "end_date": date(2025, 10, 5),
}
TEST_PERIODS = [
    {"period_no": number, "start_time": time.fromisoformat(start),
     "end_time": time.fromisoformat(end)}
    for number, start, end in (
        (1, "08:00", "08:45"), (2, "08:55", "09:40"),
        (3, "10:00", "10:45"), (4, "10:55", "11:40"),
        (5, "14:00", "14:45"), (6, "14:55", "15:40"),
        (7, "16:00", "16:45"), (8, "16:55", "17:40"),
    )
]
TEST_SCHEDULE = [
    {
        "term_id": TEST_TERM["term_id"], "week_no": week,
        "weekday": weekday, "course_name": course,
        "start_period": start, "end_period": end, "classroom": classroom,
    }
    for week in range(1, 6)
    for weekday, course, start, end, classroom in (
        (1, "硕士生公共英语", 1, 2, "教学楼201"),
        (2, "机器学习", 3, 4, "教学楼202"),
        (3, "高级工程数学", 1, 2, "教学楼102"),
        (4, "算法设计与分析", 1, 2, "教学楼102"),
        (5, "研究生学术规范与学术诚信", 5, 6, "教学楼301"),
    )
]


def _insert_missing(
    connection: Connection, table: Table, rows: list[dict], keys: tuple[str, ...],
) -> int:
    """插入缺失记录，拒绝覆盖具有相同键的不同数据。"""
    inserted = 0
    for row in rows:
        conditions = [table.c[key] == row[key] for key in keys]
        existing = connection.execute(select(table).where(*conditions)).mappings().first()
        if existing is not None:
            if any(existing[key] != value for key, value in row.items()):
                raise ValueError(f"{table.name} 已有不同内容，停止导入，未覆盖记录。")
            continue
        connection.execute(insert(table).values(**row))
        inserted += 1
    return inserted


def seed_timetable(connection: Connection) -> dict[str, int]:
    """创建三张表并补充测试记录，由调用方管理事务。"""
    Base.metadata.create_all(connection, tables=list(TABLES))
    return {
        "academic_terms": _insert_missing(
            connection, TABLES[0], [TEST_TERM], ("term_id",),
        ),
        "class_periods": _insert_missing(
            connection, TABLES[1], TEST_PERIODS, ("period_no",),
        ),
        "course_schedule": _insert_missing(
            connection, TABLES[2], TEST_SCHEDULE,
            ("term_id", "week_no", "weekday", "start_period"),
        ),
    }


def main() -> None:
    """从项目配置连接 MySQL，导入并输出新增记录数。"""
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(project_dir / ".env")
    # URL.create 处理密码中的特殊字符，且不输出连接凭据。
    url = URL.create(
        "mysql+pymysql",
        username=os.environ.get("MYSQL_USER", "root"),
        password=os.environ.get("MYSQL_PASSWORD", "root"),
        host=os.environ.get("MYSQL_HOST", "localhost"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        database=os.environ.get("MYSQL_DATABASE", "content_builder_agent"),
        query={"charset": "utf8mb4"},
    )
    engine = create_engine(url, connect_args={"connect_timeout": 5})
    try:
        with engine.begin() as connection:
            counts = seed_timetable(connection)
        print(f"数据库：{url.database}")
        print(f"学期：{TEST_TERM['term_id']}（2025-09-01 ～ 2025-10-05）")
        for table, count in counts.items():
            print(f"{table}：新增 {count} 条")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
