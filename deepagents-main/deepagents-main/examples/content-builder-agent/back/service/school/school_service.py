"""旧 SQL 服务的兼容入口，统一使用课表 AST 校验，不保留字符串表名检查。"""

from back.service.school.sql_service import SqlValue, execute_school_sql


async def execute_sql(sql: str, *, params: dict[str, SqlValue] | None = None) -> list[list]:
    """执行课表查询并保留旧接口的行列表返回形式，完整结果由 MCP 工具提供。"""
    result = await execute_school_sql(sql, params or {})
    return result["rows"]
