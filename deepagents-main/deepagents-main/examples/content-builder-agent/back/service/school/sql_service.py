"""校验并执行模型生成的课表 SQL，AST 与实际执行使用同一份规范化语句。"""

import asyncio
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, OptimizeError
from sqlglot.optimizer.scope import traverse_scope
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from back.common.exceptions import DatabaseException, ServiceException


SqlValue = str | int | float | bool | None
ALLOWED_TABLES = frozenset({"academic_terms", "course_schedule", "class_periods"})
MAX_ROWS = 100
_PARAMETER = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
# 采用允许列表，未实现检查的语法默认拒绝，而非假设所有 SELECT 都没有副作用。
_NODES = frozenset({
    exp.Select, exp.Union, exp.From, exp.Join, exp.Table, exp.TableAlias,
    exp.Identifier, exp.Column, exp.Alias, exp.Star, exp.Where, exp.Group,
    exp.Having, exp.Order, exp.Ordered, exp.Limit, exp.Offset, exp.Distinct,
    exp.Subquery, exp.With, exp.CTE, exp.Placeholder, exp.Literal,
    exp.Null, exp.Boolean, exp.And, exp.Or, exp.Not, exp.Paren,
    exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Is,
    exp.In, exp.Between, exp.Like, exp.Escape, exp.Exists,
    exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod, exp.Neg,
    exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max, exp.Coalesce,
    exp.Lower, exp.Upper, exp.Round, exp.Concat, exp.Abs,
    exp.Case, exp.If,
})


@dataclass(frozen=True)
class ValidatedQuery:
    """经过校验和行数限制的查询。"""

    sql: str
    params: dict[str, SqlValue]
    row_limit: int


def _check_tree(tree: exp.Expression) -> None:
    """拒绝未支持语法、注释、变量、递归、跨库列引用及内联字符串值。"""
    nodes = list(tree.walk())
    if len(nodes) > 600:
        raise ServiceException("SQL 过于复杂，请缩小查询范围。")
    for node in nodes:
        if type(node) not in _NODES:
            raise ServiceException(f"SQL 包含不允许的语法或函数：{type(node).__name__}。")
        if node.comments:
            raise ServiceException("SQL 不允许包含注释或优化提示。")
        if isinstance(node, exp.With) and node.args.get("recursive"):
            raise ServiceException("不允许递归 CTE。")
        if isinstance(node, exp.Literal) and node.is_string:
            raise ServiceException("字符串条件值必须用 :name 占位符，通过 params 提供。")
        if isinstance(node, exp.Column) and (node.db or node.catalog):
            raise ServiceException("列引用不允许指定数据库或 catalog。")


def _check_sources(tree: exp.Expression) -> None:
    """逐作用域检查物理表，区分 CTE/子查询别名与真正读取的表。"""
    physical = 0
    try:
        scopes = traverse_scope(tree)
        for scope in scopes:
            for _, source in scope.selected_sources.values():
                if not isinstance(source, exp.Table):
                    continue  # CTE/派生表的真实来源在其子作用域中检查
                physical += 1
                if source.db or source.catalog or source.name not in ALLOWED_TABLES:
                    raise ServiceException(
                        "只允许访问 academic_terms、course_schedule、class_periods，"
                        "禁止数据库前缀和系统表。"
                    )
                if not isinstance(source.this, exp.Identifier):
                    raise ServiceException("不允许动态表名或表函数。")
    except OptimizeError as error:
        raise ServiceException("表别名或作用域无效，请修正 SQL。") from error
    if not physical:
        raise ServiceException("查询必须实际读取允许的课表业务表。")


def _check_params(tree: exp.Expression, params: dict[str, SqlValue]) -> dict[str, SqlValue]:
    """参数仅允许 JSON 标量，占位符和参数名必须完全对应。"""
    if not isinstance(params, dict) or len(params) > 100:
        raise ServiceException("params 必须是最多100个键的参数字典。")
    names = set()
    for placeholder in tree.find_all(exp.Placeholder):
        if not _PARAMETER.fullmatch(placeholder.name):
            raise ServiceException("请使用 :name 格式的命名占位符，不使用 ? 或 %s。")
        names.add(placeholder.name)
    if any(not isinstance(key, str) or not _PARAMETER.fullmatch(key) for key in params):
        raise ServiceException("参数名必须以英文字母开头，仅含字母、数字和下划线。")
    if names != set(params):
        raise ServiceException(
            f"占位符与 params 不一致：缺少 {sorted(names - set(params))}，"
            f"多余 {sorted(set(params) - names)}。"
        )
    for value in params.values():
        if value is not None and type(value) not in (str, int, float, bool):
            raise ServiceException("参数值仅支持字符串、数字、布尔和 null，不支持列表或对象。")
        if isinstance(value, str) and len(value) > 4000:
            raise ServiceException("单个字符串参数不能超过4000字符。")
        if isinstance(value, float) and not math.isfinite(value):
            raise ServiceException("浮点参数必须是有限值。")
    return dict(params)


def _limit_rows(tree: exp.Expression) -> int:
    """所有查询层级的 LIMIT/OFFSET 都只允许有界整数。"""
    for node in tree.find_all(exp.Limit, exp.Offset):
        value = node.expression
        if not isinstance(value, exp.Literal) or not value.is_int:
            raise ServiceException("LIMIT/OFFSET 必须使用整数常量。")
        number = int(value.this)
        maximum = MAX_ROWS if isinstance(node, exp.Limit) else 10000
        if not 0 <= number <= maximum:
            raise ServiceException(f"{type(node).__name__} 不能超过 {maximum}。")
    limit = tree.args.get("limit")
    requested = int(limit.expression.this) if limit is not None else MAX_ROWS
    # 多取一条判断截断，不把不完整结果误当成全量结果。
    tree.limit(requested + 1, copy=False)
    return requested


def validate_school_sql(sql: str, params: dict[str, SqlValue]) -> ValidatedQuery:
    """将模型 SQL 解析为 MySQL AST，校验后重新生成实际执行的 SQL。"""
    if not isinstance(sql, str) or not sql.strip() or len(sql) > 12000:
        raise ServiceException("sql 必须是1～12000字符的查询文本。")
    try:
        statements = [item for item in sqlglot.parse(sql, read="mysql") if item is not None]
    except ParseError as error:
        raise ServiceException("SQL 无法按 MySQL 语法解析，请检查 SELECT、JOIN 和占位符。") from error
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.Union)):
        raise ServiceException("仅允许单条只读 SELECT 查询（可包含非递归 CTE 或 UNION）。")
    tree = statements[0]
    _check_tree(tree)
    _check_sources(tree)
    values = _check_params(tree, params)
    limit = _limit_rows(tree)
    canonical = tree.sql(dialect="mysql", comments=False)
    # 再核对 SQLAlchemy 实际识别的绑定参数，避免解析器间占位符语义差异。
    if set(text(canonical).compile().params) != set(values):
        raise ServiceException("SQL 占位符无法正确绑定，请使用简单的 :name 形式。")
    return ValidatedQuery(canonical, values, limit)


def _json_value(value: object) -> object:
    """将数据库时间、日期、小数转为工具可传输的 JSON 值。"""
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, Decimal):
        return str(value)
    return value


async def execute_school_sql(sql: str, params: dict[str, SqlValue]) -> dict:
    """以只读事务执行通过 AST 校验的 SQL，并单独绑定参数值。"""
    from back.repository.db_client import AsyncSessionLocal

    query = validate_school_sql(sql, params)
    try:
        async with asyncio.timeout(10), AsyncSessionLocal() as session:
            if session.bind.dialect.name == "mysql":
                await session.execute(text("SET TRANSACTION READ ONLY"))
            result = await session.execute(text(query.sql), query.params)
            columns = list(result.keys())
            rows = result.fetchmany(query.row_limit + 1)
            return {
                "ok": True, "columns": columns,
                "rows": [[_json_value(value) for value in row] for row in rows[:query.row_limit]],
                "row_count": min(len(rows), query.row_limit),
                "truncated": len(rows) > query.row_limit,
            }
    except TimeoutError as error:
        raise DatabaseException("数据库查询超过10秒，请缩小范围。") from error
    except DBAPIError as error:
        args = getattr(error.orig, "args", ())
        if args and args[0] in (1052, 1054, 1055, 1064, 1140, 1146):
            raise ServiceException(f"SQL 执行错误，请按表结构修正：{str(args[-1])[:300]}") from error
        raise DatabaseException("课表数据库查询失败，请检查连接或查询条件。") from error
