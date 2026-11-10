"""MCP 客户端 -- 连接基于 stdio 的 MCP 服务器，发现工具并转发工具调用。

全流程概述:
  1.【加载配置】McpManager._load_configs() 从源码内置、用户全局、项目三个位置
     读取 settings.json，合并成 {服务器名: {command, args, env}} 字典。
  2.【启动连接】使用asyncio.create_subprocess_exec启动服务器进程，配置
   stdin/stdout/stderr 三根 OS 管道，并使用asyncio.create_task启动后台协程
    _read_loop 持续监听子进程 stdout 的 JSON-RPC 响应行。
  3.【协议握手】向管道写入JSON-RPC 请求 "initialize"（带协议版本和客户端身份），
    并为对应请求id创建Future，用于后续存放从管道读取的服务器发送的内容
  4.【校验服务器】获取"initialize"请求内容后，校验版本兼容，成功则发一条
  无 id 的通知 "notifications/initialized"，服务器收到后切换为就绪状态。
  4.【发现工具】list_tools() 发 "tools/list" 请求，获取服务器提供的工具列表。
     所有服务器的工具汇总到 McpManager._tools，由 get_tool_definitions() 包装成
     mcp__{服务器名}__{工具名} 格式，注入 Agent 的工具列表。

  5.【运行时调用】LLM 调用 MCP 工具时，McpManager.call_tool() 按前缀名解析出服务器和
     工具名，找到对应的 McpConnection，发 "tools/call" 请求，返回文本结果给 LLM。
  6.【清理关闭】Agent.close() 触发 McpManager.disconnect_all() → 每个 McpConnection
     的 close() 取消后台读协程、杀死子进程、清理所有等待中的 Future。

Future 盒子机制（异步请求-响应匹配的核心）:
  _send_request 每次调用时:
    1. 分配一个自增的 req_id，往子进程 stdin 写入带该 id 的 JSON-RPC 请求
    2. 创建一个 Future 对象（类似"空盒子"），以 req_id 为 key 存入 self._pending 字典
    3. await fut 挂起，等待盒子被填满
  _read_loop 后台协程持续从子进程 stdout 读取响应行:
    1. 解析 JSON，提取响应中的 id
    2. 用 id 从 _pending 字典中取出对应的 Future（同一个引用对象）
    3. 调用 fut.set_result(...) 把服务器返回的数据"装入盒子"
    4. 盒子一旦有内容，_send_request 中 await fut 立即被唤醒，拿到结果并返回
  整个过程: 请求方创建空盒子→把盒子按编号存起来→挂起等待; 读协程收到响应→按编号找到盒子→把结果放进去→唤醒请求方。

JSON-RPC 两种写入方式:
  - _send_request:  写带 id 的 JSON-RPC 请求 → 走上述 Future 盒子机制等待响应
  - _send_notification: 写不带 id 的 JSON-RPC 通知 → 不创建 Future，不等待响应，发完即走
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════
# McpConnection -- 单个 MCP 服务器的连接
# ═══════════════════════════════════════════════════════════════

class McpConnection:
    """管理单个 MCP 服务器子进程及其 JSON-RPC 通信。

    每个 McpConnection 对应一个 MCP 服务器子进程，通过 stdin/stdout 管道
    用 JSON-RPC 2.0 协议通信。

    调用位置:
      - 由 McpManager.load_and_connect() 创建，每发现一个服务器配置就 new 一个
      - load_and_connect中按箭头依次执行：conn.connect() → conn.initialize() → conn.list_tools() → 注册到管理器
      - 运行时由 McpManager.call_tool() 调 conn.call_tool() 转发工具调用
      - Agent.close() → McpManager.disconnect_all() → conn.close() 清理
    """

    def __init__(self, server_name: str, command: str,
                 args: list[str] | None = None,
                 env: dict[str, str] | None = None):
        # 服务器标识名，对应 settings.json 里的 key（如 "serpapi"）
        self.server_name = server_name
        # 启动子进程的命令（如 "python"、"node"）
        self.command = command
        # 命令参数（如 [".claude/mcp/serpapi_server.py"]）
        self.args = args or []
        # 额外的环境变量，会合并到子进程的环境中
        self.env = env or {}

        # 子进程对象，connect() 时创建
        self._process: asyncio.subprocess.Process | None = None
        # 后台读管道的协程任务
        self._reader_task: asyncio.Task | None = None
        # JSON-RPC 请求编号，自增，用于匹配请求和响应
        self._next_id = 1
        # 等待响应的 Futur 字典，key=请求id, value=Futur
        self._pending: dict[int, asyncio.Future] = {}

    # ================mcp连接=========================
    async def connect(self) -> None:
        """启动 MCP 服务器子进程，创建 stdin/stdout/stderr 三根管道，
        并启动后台 _read_loop 持续监听子进程的输出。

        调用位置: McpManager.load_and_connect()，每个服务器配置调用一次。
        """
        # 合并父进程环境变量 + 配置文件中的额外环境变量
        merged_env = {**os.environ, **self.env}
        # 创建子进程对象:执行 command + args + 配置管道
        self._process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=subprocess.PIPE,    # OS 创建管道写端 → self._process.stdin
            stdout=subprocess.PIPE,   # OS 创建管道读端 → self._process.stdout
            stderr=subprocess.PIPE,   # 子进程的 stderr（用于调试）
            env=merged_env,           # 子进程继承的环境变量
        )
        # 把_read_loop()加入就绪队列，connect函数体执行结束。等事件循环从队列调度时开始执行，
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:

        """后台协程：监听（死循环）读管道的子进程输出，只有服务器写东西，_read_loop才能监听到内容
        1.持续从子进程 stdout 管道读取 JSON-RPC 响应行，
        2.把读取到的内容写入对应请求id对应的Furtue中

        由 connect() 启动，子进程存活期间一直运行。
        子进程退出（管道关闭）时自动结束。
        """
        assert self._process and self._process.stdout
        while True:  # 只要管道有内容，whiletrue一直循环，直到管道没内容，挂起_read_loop等着self._process.stdou
            # 父进程通过 self._process.stdout 从管道读一行
            # E:'{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05",...}}\n'
            line = await self._process.stdout.readline()
            if not line:
                break  # 管道关闭，子进程退出了
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # 非法 JSON，跳过
            # 按 id 找到结果中的Future，并当set_result时，send_request的await处接收到值
            msg_id = msg.get("id")
            if msg_id is not None and msg_id in self._pending:
                fut = self._pending.pop(msg_id)  # pop取出来的时同一个引用对象
                if "error" in msg:
                    # 服务器返回错误 → 让等待方抛异常
                    e = msg["error"]
                    fut.set_exception(RuntimeError(
                        f"MCP error {e.get('code')}: {e.get('message')}"))
                else:
                    # 正常结果 → await fut的地方被唤醒，接收到值
                    fut.set_result(msg.get("result"))

    #================mcp握手=========================
    async def initialize(self) -> None:
        """MCP 握手：协商协议版本、声明客户端身份，发送初始化完成通知。

        调用位置: McpManager.load_and_connect()，connect() 之后立即调用。
        必须先握手成功，才能调 list_tools() 和 call_tool()。
        """
        # 第一步：发送 initialize 请求，等待服务器确认
        result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",  # MCP 协议版本
            "capabilities": {},  # 客户端能力（空=基础客户端）
            "clientInfo": {  # 客户端标识
                "name": "claude-code-agent",
                "version": "1.0.0",
            },
        })
        # 第二步：校验服务器协议版本是否兼容
        server_version = result.get("protocolVersion") if isinstance(result, dict) else None
        if server_version != "2024-11-05":
            raise RuntimeError(
                f"MCP server '{self.server_name}' protocol mismatch: "
                f"expected 2024-11-05, got {server_version}")
        # 第三步：发送初始化完成通知（不需要响应）
        await self._send_notification("notifications/initialized")

    #_send_request返回的结果就是由_read_loop从管道中获取内容并写入盒子Future的
    async def _send_request(self, method: str,
                            params: dict | None = None) -> Any:
        """1.向子进程发送 JSON-RPC 请求（往管道里写了json rpc,不调用任何方法)，子进程看到管道内容自动执行
           2.通过请求id为当前请求配一个Future
           3.等盒子里的东西，等到后return
        等待readloop取出管道中对应id的内容放入盒子以供使用

        被以下方法调用:
          - initialize() → _send_request("initialize", ...)
          - list_tools()  → _send_request("tools/list")
          - call_tool()   → _send_request("tools/call", ...)

        返回: 服务器响应的 result 字段内容。
        """
        assert self._process and self._process.stdin
        # 分配请求编号
        req_id = self._next_id
        self._next_id += 1

        # 构造 JSON-RPC 请求
        msg = json.dumps({
            "jsonrpc": "2.0",          # 协议版本，固定
            "id": req_id,              # 请求编号，用于匹配响应
            "method": method,          # 要调的方法
            "params": params or {},    # 方法的参数
        })
        # 父进程通过 self._process.stdin 往管道写一行 JSON
        self._process.stdin.write((msg + "\n").encode())
        await self._process.stdin.drain()#把之前 write 积在缓冲区里的数据真正推送到管道。

        # 创建一个 Future给req_id请求，当请求有结果，_read_loop 收到响应后会用set_result把结果放入Future，await fut被唤醒，可以继续执行return
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut
        return await fut  # 挂起，等待 _read_loop 投递结果

    #向子进程写入无id的jsonrpc代表是发送通知，服务器收到通知则切换状态代表启动成功，后续可以读取工具
    async def _send_notification(self, method: str,
                                 params: dict | None = None) -> None:
        """向子进程发送 JSON-RPC 通知（无 id，不期待响应）。

        调用位置: initialize() 发送 "notifications/initialized" 通知服务器。
        """
        if not self._process or not self._process.stdin:
            return
        # 通知和请求的区别：没有 "id" 字段
        msg = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        })
        self._process.stdin.write((msg + "\n").encode())
        await self._process.stdin.drain()  # 确保数据推送到管道，防止滞留在缓冲区
    #3==================列出mcp工具========================
    async def list_tools(self) -> list[dict]:
        """向 MCP 服务器查询可用工具列表，返回标准化的工具定义。

        调用位置: McpManager.load_and_connect()中，initialize() 之后调用。
        返回的工具会被加上 serverName 字段，最终注册到 Agent 的 tools 列表。

        返回格式:
          [{"name": "web_search", "description": "...",
            "inputSchema": {...}, "serverName": "serpapi"}]
        """
        #result是server.py中response的result字段
        result = await self._send_request("tools/list")
        if not result or not isinstance(result.get("tools"), list):
            return []
        # 给每个工具加上所属服务器名，方便后续按名查找
        return [
            {"name": t["name"],
             "description": t.get("description", ""),
             "inputSchema": t.get("inputSchema"),
             "serverName": self.server_name}
            for t in result["tools"]
        ]

    #====================调用mcp工具=======================
    async def call_tool(self, name: str, args: dict) -> str:
        """向 MCP 服务器转发工具调用，返回文本结果。

        调用位置: McpManager.call_tool()，由 LLM 的工具调用触发。
        当 LLM 调 mcp__serpapi__web_search 时，经过 McpManager 解析后
        最终到达这里：call_tool("web_search", {"query": "..."})

        返回: 工具执行结果的纯文本。
        """
        result = await self._send_request(
            "tools/call",
            {"name": name, "arguments": args}
        )
        # 从 MCP 标准响应格式中提取文本内容
        # MCP 规定 content 是列表，每项有 type 和 text
        if isinstance(result, dict) and isinstance(result.get("content"), list):
            return "\n".join(
                c["text"] for c in result["content"]
                if c.get("type") == "text"
            )
        # 兜底：如果格式不标准，直接返回整个结果的 JSON
        return json.dumps(result)

    # =====================关闭mcp连接======================
    def close(self) -> None:
        """关闭连接：取消读循环协程，杀死子进程，清理所有等待中的 Future。

        调用位置:
          - McpManager.disconnect_all() → 遍历所有连接调 close()
          - McpManager.load_and_connect() 出错时调 close() 清理失败的连接
          - Agent.close() → disconnect_all() → close()
        """
        # 停止后台读循环
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        # 杀死子进程
        if self._process:
            try:
                self._process.kill()
            except ProcessLookupError:
                pass  # 子进程已经死了
            self._process = None
        # 所有等待中的请求以异常结束
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError(
                    f"MCP server '{self.server_name}' closed"))
        self._pending.clear()

# ═══════════════════════════════════════════════════════════════
# McpManager -- MCP 连接管理器
# ═══════════════════════════════════════════════════════════════

class McpManager:
    """管理所有 MCP 服务器连接。

    一个 Agent 只有一个 McpManager 实例，管理多个 McpConnection。

    调用链:
      CodingAgent.__init__ → McpManager()
      CodingAgent.chat()   → McpManager.load_and_connect()
      CodingAgent._execute_one_tool_call → McpManager.call_tool()
      CodingAgent.close()  → McpManager.disconnect_all()

    self._connections 结构示例:
      {"serpapi":  McpConnection("serpapi",  _process=子进程A),
       "postgres": McpConnection("postgres", _process=子进程B)}
    """

    def __init__(self):
        # 服务器名 → McpConnection 对象的映射，load_and_connect() 时填充
        self._connections: dict[str, McpConnection] = {}
        # 所有已发现的 MCP 工具定义列表（含 serverName 字段）
        self._tools: list[dict] = []
        # 防止重复加载的标记
        self._connected = False

    # ── 连接生命周期 ───────────────────────────────────────────

    #连接mcp服务器，经过握手成功，获取mcp所有工具
    async def load_and_connect(self) -> None:
        """读取配置文件，连接所有 MCP 服务器，发现并注册工具。

        流程:
          1. _load_configs() 读 settings.json，获取服务器（子进程）启动命令
          2.对每个服务器: connect（启动子进程 + 监听子进程） → initialize（握手：发送初始化请求+得到结果并校验+发送完成请求） → list_tools（列出每个服务器可用工具）
        """
        if self._connected:
            return  # 已经加载过了
        self._connected = True

        # configs为json文件的字典版
        configs = self._load_configs()
        if not configs:
            return

        timeout = 15.0  # 每个步骤的超时秒数
        for name, cfg in configs.items():
            # 为每个服务器创建独立的连接对象
            conn = McpConnection(
                name, cfg["command"], cfg.get("args"), cfg.get("env"))
            try:
                await conn.connect()
                # 有限时操作用asyncio.wait_for，超时抛 asyncio.TimeoutError，并且会取消里面那个协程。
                await asyncio.wait_for(conn.initialize(),         # MCP 握手
                                      timeout=timeout)
                server_tools = await asyncio.wait_for(            # 发现工具定义
                    conn.list_tools(), timeout=timeout)
                # 成功 → 注册连接和工具
                self._connections[name] = conn
                self._tools.extend(server_tools)
                print(f"[mcp] Connected to '{name}'"
                      f" -- {len(server_tools)} tools", flush=True)
            except Exception as e:
                print(f"[mcp] Failed to connect to '{name}': {e}",
                      flush=True)
                conn.close()  # 清理失败的连接

    #关闭连接
    async def disconnect_all(self) -> None:
        """关闭所有 MCP 连接，清空工具列表。

        调用位置: Agent.close() → disconnect_all()，程序退出时清理。
        """
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()
        self._tools.clear()
        self._connected = False

    # ── 工具发现与调用：graph.py中调用 ─────────────────────────────────

    #将self._tools的每一个字典完整，成为agent可识别的str
    def get_tool_definitions(self) -> list[dict]:
        """返回所有已发现的 MCP 工具定义，格式与 Agent 内置工具一致。

        调用位置: graph.py → CodingAgent.chat()，load_and_connect() 后调用，
        将结果追加到 self.tools 中。

        工具命名规则: mcp__{服务器名}__{工具名}
        如: serpapi 服务器的 web_search → mcp__serpapi__web_search
        这个前缀是 is_mcp_tool() 和 call_tool() 判断的依据。
        """
        return [
            {
                "name": f"mcp__{t['serverName']}__{t['name']}",
                "description": (t.get("description") or
                                f"MCP tool {t['name']} from {t['serverName']}"),
                "input_schema": (t.get("inputSchema") or
                                 {"type": "object", "properties": {}}),
            }
            for t in self._tools
        ]

    def is_mcp_tool(self, name: str) -> bool:
        """判断工具名是否为 MCP 工具（以 "mcp__" 开头）。

        调用位置: graph.py → CodingAgent._execute_one_tool_call，
        用于路由：MCP 工具走 mcp_client.call_tool()，其他走 execute_tool()。
        """
        return name.startswith("mcp__")

    async def call_tool(self, prefixed_name: str, args: dict) -> str:
        """解析前缀工具名，找到对应的连接，转发工具调用。

        调用位置: graph.py → CodingAgent._execute_one_tool_call，
        LLM 调 MCP 工具时触发。

        参数:
          prefixed_name: "mcp__serpapi__web_search" 格式的完整工具名
          args: LLM 生成的工具参数字典，如 {"query": "Python 3.13"}

        返回: 工具执行结果的文本。
        """
        # 拆解前缀名: mcp__serpapi__web_search → ["mcp", "serpapi", "web_search"]
        parts = prefixed_name.split("__")
        if len(parts) < 3:
            raise ValueError(f"Invalid MCP tool name: {prefixed_name}")
        server_name = parts[1]           # "serpapi"
        tool_name = "__".join(parts[2:]) # "web_search"（处理工具名可能含 __ 的情况）
        # 按服务器名找连接
        conn = self._connections.get(server_name)
        if not conn:
            raise RuntimeError(
                f"MCP server '{server_name}' not connected")
        # 转发给连接对象的 call_tool
        return await conn.call_tool(tool_name, args)

    # ── 配置文件读取 :把agent和用户项目下的settings.json转为字典──────────

    def _load_configs(self) -> dict[str, dict]:
        """读取四个位置的 MCP 配置，合并成 {服务器名: 配置} 字典。

        调用位置: load_and_connect()。

        读取顺序（后读的覆盖先读的）:
          1. 源码内置配置（agent 包自带，无需用户配置）
          2. ~/.claude/settings.json    （全局用户配置）
          3. .claude/settings.json      （项目配置）
          4. .mcp.json                   （备选项目配置）

        配置格式:
          {"mcpServers": {"serpapi": {"command": "python", "args": [...], "env": {...}}}}
        """
        merged: dict[str, dict] = {}
        # 源码内置配置（agent 包自带，始终加载）
        builtin_path = Path(__file__).parent.parent / ".claude" / "mcp" / "settings.json"
        self._merge_config_file(builtin_path, merged)
        # 项目配置（cwd/.claude/settings.json）
        project_path = Path.cwd() / ".claude" / "settings.json"
        self._merge_config_file(project_path, merged)
        # 备选项目配置（cwd/.mcp.json）
        mcp_json_path = Path.cwd() / ".mcp.json"
        self._merge_config_file(mcp_json_path, merged)
        return merged

    def _merge_config_file(self, path: Path,
                           target: dict[str, dict]) -> None:
        """读取单个 JSON 配置文件，将其中的 MCP 服务器配置合并到 target。

        调用位置: _load_configs()，对三个路径各调一次。

        配置文件中可以有两种格式:
          - 带 mcpServers 顶层 key: {"mcpServers": {"name": {...}}}
          - 直接写服务器: {"name": {...}}

        只合并包含 "command" 字段的有效服务器配置。
        """
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            # 兼容带 mcpServers 包装和不带两种格式
            servers = raw.get("mcpServers", raw)
            for name, config in servers.items():
                # 只接受有效配置（必须有 command 字段）
                if isinstance(config, dict) and "command" in config:
                    target[name] = config
        except Exception:
            pass  # 文件格式错误，静默忽略
