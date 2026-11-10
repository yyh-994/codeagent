# 个人课表 Text-to-SQL

查询链路：意图识别 → 后端补充校历/日期上下文 → 查询 Agent 生成 `sql + params`
→ FastMCP `query_school_db` → sqlglot AST 校验 → SQLAlchemy 参数绑定 → MySQL。

- 专属提示词：`back/agent/query_prompt.py`，包含表结构、关联关系、SQL示例、默认规则与错误修正。
- 查询 Agent 不加载 `school-query` Skill；旧 Skill 文件不在查询 Agent 的运行路径中。
- MCP 服务：`back/mcp/school_server.py`，只暴露一个数据库工具。
- MCP 接入：`back/agent/school_mcp.py`，使用 stdio 和 LangChain MCP 适配器。
- 校验执行：`back/service/school/sql_service.py`。
- 日期上下文：`get_sql_context()`，从数据库读取唯一学期，以北京时间计算查询范围。

## 运行

在项目根目录执行 `uv sync --group dev`，再按原方式启动后端。
后端自动发现并调用本地 MCP 服务，不需要额外开一个终端运行服务。
也可以在 MCP 客户端中配置 `uv run python -m back.mcp.school_server`，工作目录为项目根目录。
数据库连接读取项目 `.env` 中的 `MYSQL_*` 配置；示例数据仍是2025年测试学期。

工具调用示例：

```json
{
  "sql": "SELECT course_name, classroom FROM course_schedule WHERE week_no = :week AND weekday = :day ORDER BY start_period",
  "params": {"week": 3, "day": 4}
}
```

值通过 `session.execute(text(sql), params)` 绑定，模型选择查询结构。
AST 校验逐作用域检查真实物理表，只允许三张课表业务表；支持普通 JOIN、
非递归 CTE、子查询、UNION 和部分常见聚合函数，不支持任意 MySQL 语法。
不允许跨库、系统表、变量、锁、写操作、注释和未允许函数。
实际执行的是校验后由 AST 重新生成的 SQL，避免原文本与检查结构不同。
执行使用只读事务，最多10秒、最多返回100行，并报告是否截断。

旧本地查询函数保留兼容调用但不绑定到查询 Agent，旧 SQL 服务也统一通过 AST 校验。
后端的数据库账号权限仍由部署配置决定，AST 检查不等于数据库权限隔离。

## 验证

```powershell
uv run --no-sync python -m pytest back/tests/test_school_sql.py back/tests/test_timetable_intent.py back/tests/test_timetable_query.py back/tests/test_timetable.py -q
```

这些测试使用内存数据库、脚本化模型返回及本地 MCP 协议，不调用真实模型 API。
