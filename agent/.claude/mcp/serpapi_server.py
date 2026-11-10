"""MCP Server for SerpAPI web search.

全流程概述（从服务端视角看 MCP 协议）:
  MCP 客户端（mcp_client.py）通过 asyncio 启动本脚本为子进程，用 stdin/stdout
  两根 OS 管道进行 JSON-RPC 2.0 通信。服务端的生命周期是一个严格的状态机:

  状态机: 启动 → initialize → initialized → tools/list / tools/call → 关闭
           ↓          ↓             ↓                  ↓
        未握手    收到 initialize  收到 initialized   可以正常响应
                  返回协议版本      通知，标记就绪     工具请求

  1.【启动】客户端执行 "python serpapi_server.py"，子进程的 stdin/stdout/stderr
     分别连接到客户端创建的三根管道。main() 进入 for line in sys.stdin 主循环，
     阻塞等待客户端通过管道发来的 JSON-RPC 行。

  2.【初始化握手】客户端发 initialize 请求 → 服务端返回协议版本和能力声明 →
     客户端发 notifications/initialized 通知 → 服务端标记 _initialized=True。
     之后才能处理工具相关的请求。

  3.【列出工具】客户端发 tools/list 请求 → 服务端返回工具名、描述、inputSchema。
     客户端拿到后包装成 mcp__serpapi__web_search 注入 LLM 的工具列表。

  4.【调用工具】LLM 调用 mcp__serpapi__web_search → 客户端发 tools/call 请求 →
     服务端调用 _search() 执行 Google 搜索 → 将结果格式化为 Markdown 文本 →
     包装成 {"content": [{"type": "text", "text": "..."}]} 返回。

  5.【关闭】客户端杀死子进程 → sys.stdin 管道关闭 → for 循环退出 → 进程结束。

与客户端的通信格式:
  - 服务端从 sys.stdin 读一行 JSON（请求），向 sys.stdout 写一行 JSON（响应）
  - 日志全部输出到 sys.stderr，不污染 stdout 上的 JSON-RPC 通信
  - 带 id 的请求必须回复；不带 id 的（通知）不回复，continue 跳过
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error


# ═══════════════════════════════════════════════════════════════
# SerpAPI 搜索实现
# ═══════════════════════════════════════════════════════════════

def _search(query: str, num: int = 5) -> list[dict]:
    """调用 SerpAPI 执行 Google 搜索，返回结构化结果列表。

    调用位置: main() 中处理 tools/call 请求时，tool_name == "web_search" 分支。

    参数:
      query: 搜索关键词
      num:   返回结果数量（默认 5）

    返回:
      [{"title": "标题", "url": "链接", "snippet": "摘要"}, ...]
      出错时返回单条 error 格式的结果，不会抛异常。

    流程:
      1. 从环境变量 SERPAPI_KEY 获取 API 密钥（由客户端通过 settings.json 的 env 字段注入）
      2. 构造 SerpAPI 请求 URL（google 引擎，中文结果优先）
      3. HTTP GET 请求，解析 JSON 响应
      4. 提取 organic_results（自然搜索结果）的 title/link/snippet
      5. 如果有 answer_box（知识图谱/直接答案），插到结果最前面
    """
    # 从环境变量取 API 密钥，客户端在启动子进程时通过 env 参数注入
    api_key = os.environ.get("SERPAPI_KEY", "")
    if not api_key:
        return [{"title": "Error", "url": "", "snippet": "SERPAPI_KEY not set"}]

    # 构造 SerpAPI 请求参数
    params = urllib.parse.urlencode({
        "api_key": api_key,
        "q": query,
        "num": num,
        "engine": "google",   # 使用 Google 搜索引擎
        "gl": "cn",           # 国家: 中国
        "hl": "zh-CN",        # 语言: 简体中文
    })
    url = f"https://serpapi.com/search?{params}"

    try:
        # 发送 HTTP GET 请求，15 秒超时
        req = urllib.request.Request(url, headers={"User-Agent": "claude-code-agent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        # 提取自然搜索结果（Google 前十链接）
        results = []
        for r in data.get("organic_results", [])[:num]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "snippet": r.get("snippet", ""),
            })

        # 兜底：如果 SerpAPI 返回了 answer_box（知识图谱/精选答案），插到第一条
        # answer_box 通常是 Google 直接展示的答案卡片，比普通搜索结果更权威
        if data.get("answer_box"):
            ab = data["answer_box"]
            results.insert(0, {
                "title": f"[Answer] {ab.get('title', '')}",
                "url": ab.get("link", "") or url,
                "snippet": ab.get("answer", ab.get("snippet", "")),
            })

        return results if results else [{"title": "No results", "url": "", "snippet": "No organic results found."}]

    except urllib.error.HTTPError as e:
        return [{"title": f"HTTP Error {e.code}", "url": "", "snippet": str(e)}]
    except Exception as e:
        return [{"title": "Error", "url": "", "snippet": str(e)}]


# ═══════════════════════════════════════════════════════════════
# JSON-RPC stdio 协议主循环
# ═══════════════════════════════════════════════════════════════

def _log(msg: str) -> None:
    """调试日志输出到 stderr。

    重要: 必须写 stderr 而不是 stdout！
    stdout 是客户端与服务端唯一的 JSON-RPC 通信通道，任何额外输出都会破坏协议。
    stderr 被客户端接管用于调试，不会干扰协议。
    """
    print(f"[serpapi-mcp] {msg}", file=sys.stderr, flush=True)


def main() -> None:
    """MCP JSON-RPC 主循环: 从 stdin 读请求 → 处理 → 向 stdout 写响应。

    调用链:
      mcp_client.py: McpConnection.connect()
        → asyncio.create_subprocess_exec("python", "serpapi_server.py")
        → 子进程启动，执行 main()
        → for line in sys.stdin 阻塞等待客户端写入管道的数据

    通信方式:
      客户端往子进程的 stdin 管道写 JSON-RPC 请求行
      → 服务端 sys.stdin.readline() 读到
      → 处理后 sys.stdout.write(response) + flush() 写回管道
      → 客户端的 _read_loop 从管道读到响应行
    """
    _log("Server started")

    # ═════════════════════════════════════════════════════════════
    # MCP 生命周期状态机
    # ═════════════════════════════════════════════════════════════
    #
    # 状态转换:
    #   (未初始化) ──initialize──→ _initialize_done=True ──notifications/initialized──→ _initialized=True
    #
    # 两个状态标记的职责:
    #   _initialize_done: 是否已处理过 initialize 请求（防止重复 initialize）
    #   _initialized:     是否已完成完整握手（控制 tools/list 和 tools/call 的访问权限）
    #
    # 为什么分两步?
    #   MCP 协议规定: 客户端先发 initialize 请求协商版本，服务器确认后，
    #   客户端再发 initialized 通知。服务器收到通知后才算"就绪"。
    #   在就绪之前拒绝一切工具操作，保证协议的严格性。
    _initialized = False       # 是否已完成初始化通知（收到 notifications/initialized）
    _initialize_done = False   # 是否已发过 initialize 响应（收到过 initialize 请求）

    # ── 主循环: 逐行从 stdin 管道读取 JSON-RPC ──
    # for 循环在 sys.stdin 上迭代: 每次迭代阻塞等待管道中的一行数据
    # 当客户端关闭管道（子进程被杀或客户端断开）时，循环自动退出
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue  # 跳过空行

        # 解析 JSON-RPC 请求
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _log(f"Invalid JSON: {line[:100]}")
            continue  # 非法 JSON，跳过（不回复，因为没有有效 id）

        # 提取 JSON-RPC 核心字段
        req_id = request.get("id")          # 请求编号（通知没有 id）
        method = request.get("method", "")   # 方法名
        params = request.get("params", {})   # 方法参数

        # response 在下面各分支中赋值，最后统一 write 到 stdout
        response = None

        # ══════════════════════════════════════════════════════
        # 分支 1: initialize — 协议握手第一步
        # ══════════════════════════════════════════════════════
        if method == "initialize":#校验版本请求
            if _initialize_done:
                # 不允许重复初始化，每个连接只能握一次手
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": "Already initialized"},
                }
                _log(f"Rejected duplicate initialize (id={req_id})")
            else:
                _log(f"Received initialize (id={req_id})")
                _initialize_done = True
                # 返回协议版本和服务器能力
                # capabilities.tools={} 表示本服务器支持工具调用
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",  # MCP 协议版本，需与客户端一致
                        "capabilities": {"tools": {}},     # 声明支持的工具能力
                        "serverInfo": {"name": "serpapi-web-search", "version": "1.0.0"},
                    },
                }

        # ══════════════════════════════════════════════════════
        # 分支 2: 通知处理 — 不带 id，不回复
        # ══════════════════════════════════════════════════════
        elif method.startswith("notifications/"):#握手成功发送通知请求
            if method == "notifications/initialized":
                # initialized 通知: 客户端告知握手完成，服务端标记就绪
                if not _initialize_done:
                    _log("Rejected: initialized notification before initialize")
                elif _initialized:
                    _log("Rejected: duplicate initialized notification")
                else:
                    _initialized = True
                    _log("Connection initialized ✓")
            else:
                _log(f"Ignoring unknown notification: {method}")
            continue  # 通知一律不回复（JSON-RPC 规范: 无 id 的请求无需响应）

        # ══════════════════════════════════════════════════════
        # 守卫: 以下所有方法都需要先完成握手
        # ══════════════════════════════════════════════════════
        elif not _initialized:
            _log(f"Rejected {method}: not initialized (id={req_id})")
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32001, "message": "Not initialized. Send initialize request first."},
            }

        # ══════════════════════════════════════════════════════
        # 分支 3: tools/list — 返回可用工具列表
        # ══════════════════════════════════════════════════════
        elif method == "tools/list":#获取工具列表请求
            _log(f"Received tools/list (id={req_id})")
            # 工具定义包含三个核心字段:
            #   name:        工具名（客户端会加 mcp__serpapi__ 前缀）
            #   description: 工具描述（LLM 根据描述决定是否调用）
            #   inputSchema: 参数 JSON Schema（LLM 根据 schema 生成参数）
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "web_search",
                            "description": (
                                "Search the web using Google. "
                                "Returns titles, URLs, and snippets for each result. "
                                "Use this to find current information, documentation, "
                                "or answers to questions."
                            ),
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The search query",
                                    },
                                    "num": {
                                        "type": "number",
                                        "description": "Number of results to return (default 5, max 10)",
                                    },
                                },
                                "required": ["query"],  # query 是必填参数
                            },
                        }
                    ]
                },
            }

        # ══════════════════════════════════════════════════════
        # 分支 4: tools/call — 执行工具调用
        # ══════════════════════════════════════════════════════
        elif method == "tools/call":#调用工具请求
            tool_name = params.get("name", "")          # 要调的工具名
            arguments = params.get("arguments", {})      # LLM 生成的参数
            _log(f"Received tools/call name={tool_name} (id={req_id})")

            if tool_name == "web_search":
                # 从 LLM 参数中提取搜索词和数量
                query = arguments.get("query", "")
                num = min(int(arguments.get("num", 5)), 10)  # 最多 10 条，防止滥用
                _log(f"Searching: {query} (num={num})")

                # 调用 SerpAPI 执行搜索
                results = _search(query, num)

                # 将结果格式化为 Markdown 文本
                # 每个结果: 标题 + URL + 摘要 + 空行
                lines = []
                for i, r in enumerate(results):
                    lines.append(f"## {i + 1}. {r['title']}")
                    lines.append(f"URL: {r['url']}")
                    lines.append(f"{r['snippet']}")
                    lines.append("")  # 空行分隔
                text = "\n".join(lines)

                # MCP 标准响应格式: result.content 是列表，每项有 type 和 text
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": text}],
                    },
                }
            else:
                # 未知工具 → 返回错误
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                }

        # ══════════════════════════════════════════════════════
        # 分支 5: 未知方法
        # ══════════════════════════════════════════════════════
        else:
            _log(f"Unknown method: {method}")
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }

        # ── 统一响应出口 ──
        # 所有分支构造好 response 后，在这里写一行 JSON 到 stdout 管道
        # ensure_ascii=False 保证中文不被转义为 \uxxxx
        # flush() 确保立即推送到管道，不滞留在 Python 缓冲区
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    # 循环退出 = stdin 管道关闭 = 客户端断开了连接或杀死了子进程
    _log("Server stopped")


if __name__ == "__main__":
    main()
