"""课表 FastMCP 服务：python -m back.mcp.school_server，以 stdio 通信。"""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from back.common.exceptions import AppException
from back.service.school.sql_service import execute_school_sql


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """每次 MCP 进程退出时释放数据库连接池。"""
    from back.repository.db_client import engine
    try:
        yield {}
    finally:
        await engine.dispose()


mcp = FastMCP("school-timetable", lifespan=lifespan)


@mcp.tool()
async def query_school_db(
    sql: str, params: dict[str, str | int | float | bool | None],
) -> dict[str, object]:
    """执行模型生成的课表 MySQL SELECT SQL 模板，参数值单独放在 params。

    仅允许 academic_terms、course_schedule、class_periods，不允许数据库前缀或系统表。
    sql 中用 :name 占位符，不能加引号；params 必须提供所有且仅有对应参数。
    支持单表、多表JOIN、非递归CTE、子查询和UNION。最多返回100行。
    后端用sqlglot检查AST、SQLAlchemy绑定参数执行。失败返回ok=false和error用于修正。
    """
    try:
        return await execute_school_sql(sql, params)
    except AppException as error:
        return {"ok": False, "error": error.message, "code": error.code}


if __name__ == "__main__":
    # 仅启动独立服务时读取配置，导入模块/单元测试不修改进程环境。
    # 数据库客户端由 lifespan/查询函数延迟导入，因此仍先于连接池创建加载。
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    mcp.run(transport="stdio")
