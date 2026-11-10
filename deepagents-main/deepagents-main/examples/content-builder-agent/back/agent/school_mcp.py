"""发现课表 FastMCP 工具并转换为 DeepAgents 可调用的工具。"""

import asyncio
import sys
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


_tools: list[BaseTool] | None = None
_lock = asyncio.Lock()


async def get_query_tools() -> list[BaseTool]:
    """首次从本地 stdio MCP 服务发现工具，之后复用无会话工具适配器。"""
    global _tools
    async with _lock:
        if _tools is None:
            project_dir = Path(__file__).resolve().parents[2]
            client = MultiServerMCPClient({
                "school": {
                    "transport": "stdio", "command": sys.executable,
                    "args": ["-m", "back.mcp.school_server"],
                    "cwd": str(project_dir),
                },
            })
            async with asyncio.timeout(20):
                tools = await client.get_tools()
            if len(tools) != 1 or tools[0].name != "query_school_db":
                raise RuntimeError("课表 MCP 服务必须仅暴露 query_school_db 工具。")
            _tools = tools
    return list(_tools)


def loaded_query_tools() -> list[BaseTool]:
    """同步工厂取已发现的 MCP 工具，初始化由请求编排或服务启动完成。"""
    if _tools is None:
        raise RuntimeError("请先 await get_query_tools() 初始化课表 MCP 工具。")
    return list(_tools)
