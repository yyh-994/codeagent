"""LangGraph agent -- 核心图引擎，用声明式节点和边替代传统的 while 循环。

架构:
  START → preprocess → agent → router → tools → agent
                                  → END

每个节点都是纯函数：读取 state 并返回部分更新。
图引擎负责处理循环、条件路由和状态管理。

支持 Anthropic (ChatAnthropic) 和 OpenAI 兼容 (ChatOpenAI) 两种后端，
通过统一的 LangChain chat model 接口进行调用。
"""

from __future__ import annotations

import asyncio          # 异步 I/O：并发工具执行、MCP 连接
import json
from dataclasses import asdict
import os               # 环境变量：API key、base URL
import random           # 随机数：重试退避的抖动（避免惊群）
import subprocess       # 子进程：git 自动初始化/提交
import sys              # 系统：stdout 流式输出
import time             # 时间戳：会话开始时间、API 调用间隔
import uuid             # UUID：生成唯一会话 ID
from functools import reduce  # reduce：合并 LangChain 流式 chunk
from pathlib import Path      # Path：文件路径操作
from typing import Any, Callable, Awaitable  # 类型标注

# LangGraph 核心：状态图和结束标记
from langgraph.graph import StateGraph, END
# interrupt: 暂停图执行等待用户确认；Command: 恢复执行时传入的指令
from langgraph.types import interrupt, Command

# 持久化 checkpoint：优先 SQLite（生产），回退内存（开发/测试）
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError:  # pragma: no cover — 未安装 sqlite 依赖时回退
    AsyncSqliteSaver = None  # type: ignore[assignment]
from langgraph.checkpoint.memory import MemorySaver

# LangChain 消息类型 —— 图内消息流的原子单位
from langchain_core.messages import (
    HumanMessage,   # 用户输入
    AIMessage,      # LLM 回复（含 tool_calls）
    ToolMessage,    # 工具执行结果
    SystemMessage,  # 系统提示词
    BaseMessage,    # 基类
)

# 工具系统
from .state import AgentStateDict
from .tools import (
    tool_definitions,       # 内置工具定义列表（read/write/edit/grep/bash...）
    execute_tool,           # 执行单个工具调用的底层函数
    check_permission,       # 权限校验（根据 mode 决定 deny/allow/ask）
    CONCURRENCY_SAFE_TOOLS, # 可安全并发执行的工具名集合（如 read_file、grep）

    truncate_result,        # 截断过长结果，保留首尾
)
# 记忆系统
from .memory import (
    MemoryStore,
    format_memories_for_injection,
)
# 自主循环（/goal）
from .autonomy import (
    goal_directive,           # /goal 启动指令模板
    GOAL_EVALUATOR_SYSTEM,    # 裁判 LLM 的 system prompt
    GOAL_TRANSCRIPT_FRAMING,  # 裁判 LLM 的对话记录框架
    goal_judge_user_message,  # 裁判 LLM 的提问模板
    parse_goal_verdict,       # 解析裁判 LLM 的判定结果
    GOAL_MAX_ITERATIONS,      # goal 最大迭代次数
)
# UI 输出
from .ui import (
    print_assistant_text,  # 打印 LLM 回复
    print_tool_call,       # 打印工具调用信息
    print_tool_result,     # 打印工具执行结果
    print_confirmation,    # 打印确认提示
    print_divider,         # 打印分隔线
    print_cost,            # 打印费用统计
    print_retry,           # 打印重试信息
    print_info,            # 打印通用信息
    print_sub_agent_start, # 子 agent 开始标记
    print_sub_agent_end,   # 子 agent 结束标记
    start_spinner,         # 启动加载动画
    stop_spinner,          # 停止加载动画
)
# 会话持久化
from .session import save_session, load_session, get_latest_session_id
# 提示词构建
from .prompt import (
    build_static_system_prompt,   # 静态部分：规则、工具说明
    build_startup_system_context, # 启动上下文：环境、项目说明
    build_plan_mode_prompt,       # plan 模式附加指令
)
# MCP 协议客户端
from .mcp_client import McpManager
# 子 agent 类型配置（agent 工具用）
from .subagent import get_sub_agent_config


# ═══════════════════════════════════════════════════════════════
# 重试机制
# ═══════════════════════════════════════════════════════════════

def _is_retryable(error: Exception) -> bool:
    """判断是否为可重试的瞬态错误（429 限流、503/529 服务不可用、网络中断）。"""
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status in (429, 503, 529):  # 429=限流, 503=服务不可用, 529=过载
        return True
    msg = str(error)
    # 字符串匹配兜底：Anthropic 过载或网络中断
    return any(kw in msg for kw in ("overloaded", "ECONNRESET", "ETIMEDOUT"))


async def _with_retry(fn, max_retries: int = 3, abort_check: Callable[[], bool] | None = None):
    """带指数退避的重试包装器。

    每次重试等待时间 = min(1000 * 2^attempt, 30000)ms + 随机抖动[0,1)s。
    如果 abort_check 返回 True，立即抛出 CancelledError 中止重试。
    """
    for attempt in range(max_retries + 1):  # 0..3，共最多 4 次尝试
        if abort_check and abort_check():    # 用户按了 Ctrl+C → 立即中止
            raise asyncio.CancelledError("Aborted during retry")
        try:
            return await fn()               # 执行实际的 LLM 调用
        except asyncio.CancelledError:
            raise                            # 取消异常直接传播，不重试
        except Exception as error:
            if abort_check and abort_check():  # 重试间隙再次检查中止信号
                raise asyncio.CancelledError("Aborted during retry")
            if attempt >= max_retries or not _is_retryable(error):  # 已无重试次数 或 不可重试
                raise                        # 直接抛出原始异常
            # 指数退避 （避免多个客户端同时重试造成惊群）
            delay = min(1000 * (2 ** attempt), 30000) / 1000 + random.random()
            status = getattr(error, "status_code", None) or getattr(error, "status", None)
            reason = f"HTTP {status}" if status else (getattr(error, "code", None) or "network error")
            print_retry(attempt + 1, max_retries, reason)  # 打印重试提示
            await asyncio.sleep(delay)      # 等待退避时间后重试


# ═══════════════════════════════════════════════════════════════
# 上下文压缩常量
# ═══════════════════════════════════════════════════════════════

# 被截断内容的占位符，提示模型可重新读取
SNIP_PLACEHOLDER = "[Content snipped - re-read if needed]"
# 上下文利用率达到 60% 时触发截断
SNIP_THRESHOLD = 0.60
# 缓存热期（距上次 API 调用 < 5 分钟）内，利用率超过 75% 才触发截断
SNIP_HOT_OVERRIDE = 0.75
# 微压缩空闲阈值：5 分钟无 API 调用则清理旧工具结果
MICROCOMPACT_IDLE_S = 5 * 60
# 压缩时保留最近 N 条工具结果
KEEP_RECENT_RESULTS = 3


# ═══════════════════════════════════════════════════════════════
# CodingAgent -- 核心 Agent 类
# ═══════════════════════════════════════════════════════════════

class CodingAgent:
    """基于 LangGraph 的编程 Agent。

    持有 LLM 客户端和配置，构建 agent 图，对外提供简洁的 chat() 接口。

    图结构:
      START → preprocess → agent → router → tools → agent
                                       → END

    preprocess: 记忆预取、上下文压缩、注入 system prompt
    agent:      调用 LLM 生成回复或工具调用
    router:     判断是继续调工具还是结束
    tools:      权限校验 + 实际执行工具
    """

    # ── 模型能力配置（静态常量，与实例无关）──────────────
    # 上下文窗口大小（token）
    _MODEL_CONTEXT: dict[str, int] = {
        "claude-opus-4-6": 200000, "claude-sonnet-4-6": 200000,
        "claude-sonnet-4-20250514": 200000, "claude-haiku-4-5-20251001": 200000,
        "claude-opus-4-20250514": 200000, "gpt-4o": 128000, "gpt-4o-mini": 128000,
        "deepseek-chat": 128000,
    }

    # 支持自适应思考的模型（Claude 4.x 系列）
    _ADAPTIVE_THINKING_MODELS = frozenset({
        "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001",
        "claude-opus-4-20250514", "claude-sonnet-4-20250514",
    })

    # 各模型的最大输出 token 数
    _MAX_OUTPUT_TOKENS: dict[str, int] = {
        "claude-opus-4-6": 64000,
        "claude-sonnet-4-6": 32000,
        "claude-haiku-4-5-20251001": 32000,
        "claude-opus-4-20250514": 32000,
        "claude-sonnet-4-20250514": 32000,
        "gpt-4o": 16384,
        "gpt-4o-mini": 16384,
        "deepseek-chat": 8192,
    }

    def __init__(
        self,
        *,
        permission_mode: str = "default",   # 权限模式：default/plan/bypassPermissions，主 agent 和子 agent 均可继承
        modelname: str = "claude-sonnet-4-6",   # 传给 API 的模型名，主/子 agent 均可继承
        api_base: str | None = None,        # OpenAI 兼容 API 地址（仅主 agent 使用）
        anthropic_base_url: str | None = None,  # Anthropic API 自定义地址
        api_key: str | None = None,         # API 密钥（仅主 agent 使用）
        thinking: bool = False,             # 是否开启扩展思考（仅主 agent 使用）
        max_cost_usd: float | None = None,  # 单次会话成本上限（美元），超出停止（仅主 agent）
        max_turns: int | None = None,       # 单次消息最多调几轮工具，超出停止（仅主 agent）
        confirm_fn: Callable[[str], Awaitable[bool]] | None = None,  # 危险操作确认回调（非 CLI 模式使用）
        custom_system_prompt: str | None = None,  # 自定义 system prompt（子 agent 使用）
        custom_tools: list[dict] | None = None,   # 自定义工具列表（子 agent 使用）
        is_sub_agent: bool = False,         # 是否为子 agent
        resume: bool = False,               # 是否从磁盘恢复上次会话（仅主 agent，--resume 用）
    ):
        # ═══════════════════════════════════════════════════════════
        # 图外基础设施（不可序列化 / 图执行期间不变）
        # ═══════════════════════════════════════════════════════════

        # ── 模型 & API ──
        self.model_name = modelname                         # 传给 API 的模型名
        self.use_openai = bool(api_base)                # 是否走 OpenAI 兼容接口
        self._is_anthropic = not self.use_openai        # 是否走 Anthropic 原生接口
        self.thinking = thinking                        # 是否开启扩展思考
        self.effective_window = self._MODEL_CONTEXT.get(modelname, 200000) - 20000  # 有效上下文窗口（预留 2 万给输出）
        self.confirm_fn = confirm_fn                    # （非 CLI 模式）：当把 agent 当作库嵌入到 GUI / Web 服务 / 非终端环境时，无法用 input() 阻塞询问，这时由调用方传入自己的确认回调

        # ── System prompt ──
        if custom_system_prompt:
            self._static_system_prompt = custom_system_prompt
            self._startup_system_context = ""
        else:
            self._static_system_prompt = build_static_system_prompt()
            self._startup_system_context = build_startup_system_context()
        self._base_system_prompt = (
            self._static_system_prompt + "\n\n" + self._startup_system_context
            if self._startup_system_context else self._static_system_prompt
        )

        # ── LLM 客户端 & 工具 ──
        self._llm = self._create_llm(api_base, anthropic_base_url, api_key)
        self.tools = custom_tools or tool_definitions    # 主 agent 用 tool_definitions，子 agent 用 custom_tools

        # ── MCP ──
        self._mcp_manager = McpManager()
        self._mcp_initialized = False                   # 是否已初始化 MCP 连接

        # ── memory ──
        self.project_dir = Path.cwd().resolve()
        self._memory_store = None if is_sub_agent else MemoryStore(self.project_dir)
        self._memory_context = ""
        self._memory_query = ""
        self._memory_message_id: str | None = None
        self._memory_dirty = False

        # ── Goal ──
        self.active_goal: dict | None = None            # 当前活跃的 goal（condition/iterations/started_at）
        self.goal_stop = False                          # 是否收到停止 goal 的信号

        # ── 中断 & 取消，由REPL处理 ──
        self._current_task: asyncio.Task | None = None  # 当前异步任务引用（用于 cancel）
        self._interrupt_pending: dict | None = None     # 待处理的中断数据
        self._interrupt_config: dict | None = None      # 中断恢复所需的 thread config

        # ── 一次性标记 ──
        self._git_initialized: bool = False             # Git 仓库初始化标记，防重复

        # ── 终端流式输出 UI ──
        self._stream_text_started = False               # astream时使用。是否是第一个chunk，若是，则设为True，思考动画后加换行，后续 chunk 不再重复换行
        self._streaming_emitted = False                 # 流式是否输出了文字，若是，则为True，否则为空流，调用ainvoke

        # ── 上下文（每一轮chat后都要更新） ──
        self._conversation_messages: list[BaseMessage] = []

        # ── 恢复上次会话（--resume）──
        if resume:
            self.resume_latest()

        # ── 构建并编译图 ──
        self._graph = self._build_graph()

        # ── 控制 ──
        self._aborted = False  # 是否已触发中止（Ctrl+C）

        # ═══════════════════════════════════════════════════════════
        # 图内状态 — 每次 chat() 复制进 initial_state，节点通过 state 读写
        # 对应 state.py 的 AgentStateDict
        # ═══════════════════════════════════════════════════════════

        # ── 权限 ──
        self.permission_mode = permission_mode          # default / bypassPermissions / plan
        self.plan_file_path: str | None = None         # plan 模式下计划文件路径
        if permission_mode == "plan":
            self.plan_file_path = self.generate_plan_file_path()
        self.pre_plan_mode: str | None = None          # 进入 plan 前的权限模式，退出时恢复

        # ── 预算 ──
        self.max_cost_usd = max_cost_usd                # 费用上限（美元）
        self.max_turns = max_turns                      # 每轮最大工具调用轮数
        self.current_turns = 0  # 当前消息已执行的工具调用轮数

        # ── 子 agent ──
        self.is_sub_agent = is_sub_agent
        self._output_buffer: list[str] | None = None    # 子 agent 流式输出收集器

        # ── 会话 ──
        self.session_id = uuid.uuid4().hex[:8]          # 8 位随机会话 ID
        self.session_start_time = time.strftime("%Y-%m-%dT%H:%M:%S+08:00", time.localtime())

        # ── Token 统计 ──
        self.total_input_tokens = 0 #整个会话
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0  # Anthropic 缓存读取
        self.total_cache_creation_tokens = 0  # Anthropic 缓存写入
        self.last_llm_call_tokens = 0  # 上一轮 LLM 调用的 input+output是的token，用于每轮消息压缩判断
        self.last_llm_call_time = 0.0  # 上次 LLM 调用时间戳，用于压缩间隔判断

        # ── 文件追踪 ──
        self._read_file_state: dict[str, float] = {}    # 已读文件路径 → mtime，编辑前校验



    # ═════════════════════════════════════════════════════════
    # 公开 API
    # ═════════════════════════════════════════════════════════

    async def chat(self, user_message: str) -> None:
        """主入口：通过 agent 图处理用户消息。

        流程:
        1. 处理上一轮未解决的中断（生成新 session_id）
        2. 懒加载 MCP 连接
        3. 自动初始化 git 仓库
        4. 构建初始状态
        5. 流式执行图（astream with values mode）
        6. 保存会话、自动 git commit

        Args:
            user_message: 用户输入的文本
        """

       # ── 兜底：若continue_with回复是出现未知异常会冒泡到_safe_chat 的 except Exception，而没有处理异常，此时残留，需重开会话
        if self._interrupt_pending:
            self.session_id = uuid.uuid4().hex[:8]  # 生成新的 8 位 session ID
            self._interrupt_pending = None
            self._interrupt_config = None

        # ── 首次调用（mcp未初始化且非子agent）进行mcp连接 ──
        # ── 获取mcp工具拼接到工具列表self.tools ──
        if not self._mcp_initialized and not self.is_sub_agent:
            self._mcp_initialized = True  # 标记已尝试初始化，防止重复
            try:
                await self._mcp_manager.load_and_connect()  # 连接MCP服务器（创建子进程、监听子进程、发送请求3次握手、获取mcp工具）
                mcp_defs = self._mcp_manager.get_tool_definitions()  # 工具定义格式化
                if mcp_defs:
                    self.tools = self.tools + mcp_defs  # 将 MCP 工具合并到工具列表
            except Exception as e:
                print(f"[mcp] Init failed: {e}", flush=True)

        # ── 自动初始化 git 仓库（仅首次，非子 agent）──
        if not self._git_initialized and not self.is_sub_agent:
            self._git_initialized = True  # 标记已尝试，防止重复
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--git-dir"],  # 检测是否已是 git 仓库
                    capture_output=True, encoding="utf-8", timeout=3
                )
                if result.returncode != 0:
                    # 当前目录不是 git 仓库，执行 git init + 初始提交
                    subprocess.run(["git", "init"], capture_output=True, timeout=5)
                    subprocess.run(["git", "add", "."], capture_output=True, timeout=10)
                    subprocess.run(
                        ["git", "commit", "-m", "init: 项目初始化"],
                        capture_output=True, timeout=5
                    )
                    print_info("Git repository auto-initialized.")
            except Exception:
                pass

        # ── 构建消息列表（已有对话续接 > 全新对话）──
        if self._conversation_messages:
            messages = list(self._conversation_messages) + [HumanMessage(content=user_message)]
        else:
            messages = [HumanMessage(content=user_message)]  # 全新对话

        # ── 构建初始状态（LangGraph 图的输入字典）──
        initial_state: AgentStateDict = {
            "messages": messages,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_creation_tokens": 0,
            "last_llm_call_tokens": 0,
            "max_cost_usd": self.max_cost_usd,  # 传递预算配置到图内
            "max_turns": self.max_turns,
            "current_turns": 0,
            "permission_mode": self.permission_mode,
            "plan_file_path": self.plan_file_path,  # plan 模式计划文件路径
            "read_file_state": {},  # 已读文件状态（mtime 追踪）
            "session_id": self.session_id,
            "session_start_time": self.session_start_time,
            "last_llm_call_time": 0.0,
            "is_sub_agent": self.is_sub_agent,
            "output_buffer": None,
        }

        # 重置初始变量
        self._interrupt_pending = None
        self._aborted = False  # 重置中止标志
        self._current_task = asyncio.current_task()  # 获取当前 asyncio Task，供 abort() 取消
        self.current_turns = 0  # 重置本轮工具调用轮数（避免跨轮累计，--max-turns 按单条消息计）
        # 构建thread_config：session_id 作为 thread_id 实现 checkpoint 持久化
        thread_config = {"configurable": {"thread_id": self.session_id}}
        # 构建self._interrupt_config
        self._interrupt_config = thread_config  # 保存 config 供 continue_with 恢复

        # 执行图（处理中断、终止、清除、保存消息列表）
        _final_messages = messages[:]  # 保存最终消息作为上下文（初始为输入消息）
        try:
            # 以 values 模式流式执行图，每个事件是一个state
            async for event in self._graph.astream(initial_state, thread_config, stream_mode="values"):
                if "messages" in event:
                    _final_messages = event.get("messages", _final_messages)  # 更新最终消息
        except asyncio.CancelledError:
            self._aborted = True  # 用户 Ctrl+C 触发的中止，asyncio Task 被 cancel，标记中止
        finally:
            self._current_task = None  # 清除任务引用

        # langgraph 1.x 的 interrupt() 不再向调用方抛 GraphInterrupt，而是把中断写进
        # checkpoint；astream 正常结束后需主动读 get_state 的 interrupts 来发现中断。
        # 若图被暂停（plan 审批等），保存中断状态供外部处理并提前返回，不保存会话、不 commit。
        if not self._aborted:
            try:
                _st = self._graph.get_state(thread_config)
            except Exception:
                _st = None
            if _st is not None and getattr(_st, "interrupts", None):
                self._interrupt_pending = _st.interrupts[0].value
                self._interrupt_config = thread_config
                return

        # ── 对话正常结束后（仅主 agent）：保存上下文、自动提交git──
        if not self.is_sub_agent:
            self._conversation_messages = list(_final_messages)  # 缓存消息供后续轮次使用

            # 自动提交 git 改动（如果工作目录有变更）
            try:
                status = subprocess.run(
                    ["git", "status", "--short"],  # 检查是否有未提交改动
                    capture_output=True, encoding="utf-8", timeout=3
                )
                if status.stdout.strip():
                    subprocess.run(
                        ["git", "add", "."],  # 暂存所有改动
                        capture_output=True, timeout=10
                    )
                    # 用用户消息的前 80 字符作为 commit message
                    msg = user_message[:80] if len(user_message) <= 80 else user_message[:77] + "..."
                    subprocess.run(
                        ["git", "commit", "-m", msg],
                        capture_output=True, timeout=5
                    )
                    print_info(f"Auto committed: {msg}")
            except Exception:
                pass

            print_divider()
            self._session_save()  # 持久化会话到磁盘

    async def continue_with(self, resume_value: Any = None) -> None:
        """恢复被 interrupt() 暂停的图执行。

        在 chat() 因 interrupt 返回后，调用者展示确认/审批提示，
        收集用户响应，然后调用此方法恢复执行。

        Args:
            resume_value: 传给 interrupt() 内部的返回值。
                         确认场景: {"confirmed": True/False}
                         Plan 审批: {"choice": "执行/继续计划/清除后执行", ...}
        """
        if not self._interrupt_pending:  # 无待处理中断，直接返回（幂等保护）
            return

        config = self._interrupt_config  # 保存的 thread config（含 session_id）
        self._interrupt_pending = None  # 清除中断状态
        self._interrupt_config = None
        self._current_task = asyncio.current_task()  # 供 abort() 取消
        _final_messages: list = []
        try:
            # 通过 Command(resume=...) 恢复图执行：LangGraph 从 checkpoint 恢复并继续，event为每个节点运行完的完整satate
            # interrupt() 那个调用点会返回这个 value
            async for event in self._graph.astream(
                Command(resume=resume_value), config, stream_mode="values"
            ):
                if self._aborted:
                    break
                if "messages" in event:
                    _final_messages = event.get("messages", _final_messages)
        except asyncio.CancelledError:
            self._aborted = True
        finally:
            self._current_task = None

        # 恢复执行后可能又遇到新的 interrupt（嵌套确认）——从 checkpoint 读取
        if not self._aborted:
            try:
                _st = self._graph.get_state(config)
            except Exception:
                _st = None
            if _st is not None and getattr(_st, "interrupts", None):
                self._interrupt_pending = _st.interrupts[0].value
                self._interrupt_config = config
                return

        # 恢复执行完毕后的清理（仅主 agent，非中断状态）
        if not self.is_sub_agent:
            if _final_messages:
                self._conversation_messages = list(_final_messages)
            else:
                # 回退：如果流式未产生消息，从 checkpoint state 中获取最终消息
                try:
                    state = self._graph.get_state(config)  # 从持久化存储读取最新状态
                    if state and state.values:
                        self._conversation_messages = list(state.values.get("messages", []))
                except Exception:
                    pass
            print_divider()
            self._session_save()  # 持久化会话

    def is_interrupted(self) -> bool:
        """检查图是否正在等待 interrupt 审批。"""
        return self._interrupt_pending is not None

    def abort(self) -> None:
        """中止当前执行：设置标志 + 丢弃中断 + 取消异步任务。

        统一的中止入口：无论当前在跑图、等 interrupt 审批、还是空闲，
        Ctrl+C 都走这里。丢弃 _interrupt_pending 让 interrupt 审批等待
        也能被"取消"（否则图会一直等着 continue_with 恢复）。
        """
        self._aborted = True
        self._interrupt_pending = None  # 丢弃未处理的中断（审批等待时即"取消审批"）
        self._interrupt_config = None
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    def clear_history(self) -> None:
        """清除对话历史和所有统计信息。"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_creation_tokens = 0
        self.last_llm_call_tokens = 0
        self.current_turns = 0
        self._conversation_messages = []
        self._memory_context = ""
        self._memory_query = ""
        self._memory_message_id = None
        self._memory_dirty = False
        print_info("Conversation cleared. Session will restart on next message.")

    def show_cost(self) -> None:
        """打印当前费用和 token 统计。"""
        total = self._get_current_cost_usd()
        budget_info = f" / ${self.max_cost_usd} budget" if self.max_cost_usd else ""
        turn_info = f" | Turns: {self.current_turns}/{self.max_turns}" if self.max_turns else ""
        print_info(f"Tokens: {self.total_input_tokens} in / {self.total_output_tokens} out\n  Estimated cost: ${total:.4f}{budget_info}{turn_info}")

    def list_memories(self):
        """List memories from the startup directory's database."""
        return self._memory_store.list_memories() if self._memory_store else []

    async def reindex_memories(self) -> int:
        count = await self._memory_store.reindex() if self._memory_store else 0
        self._memory_dirty = True
        return count

    async def compact(self) -> None:
        """手动触发对话压缩。"""
        print_info("Manually compacting conversation...")
        if self._conversation_messages and len(self._conversation_messages) >= 5:
            await self._compact_conversation(self._conversation_messages)
        else:
            # 消息不足 5 条：在下一次轮次中强制触发压缩
            self.last_llm_call_tokens = self.effective_window

    async def close(self) -> None:
        """清理资源：断开 MCP 连接 + 关闭 SQLite checkpoint。"""
        if self._memory_store is not None:
            await asyncio.to_thread(self._memory_store.close)
        if self._mcp_initialized:
            await self._mcp_manager.disconnect_all()
        # 关闭 SQLite checkpoint 的异步上下文管理器
        cp = getattr(self, "_checkpointer", None)
        if cp is not None and hasattr(cp, "_exit_ctx"):
            try:
                await cp._exit_ctx.__aexit__(None, None, None)
            except Exception:
                pass

    # ═════════════════════════════════════════════════════════
    # 构建图
    # ═════════════════════════════════════════════════════════

    def _build_graph(self):
        """构建 Agent StateGraph。

        节点:
          preprocess  -- 记忆召回 + 上下文压缩
          agent       -- LLM 调用（生成回复或工具调用）
          tools       -- 权限检查 + 工具执行

        边:
          START → preprocess → agent → router → tools → agent
                                              → END

        使用 SQLite构建checkpoint 持久化状态以支持中断恢复，失败时回退到内存存储。
        """
        workflow = StateGraph(AgentStateDict)  # 以 TypedDict 为 schema 创建图

        # 注册三个节点
        workflow.add_node("preprocess", self._preprocess_node)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tools_node)

        # 设置入口和边
        workflow.set_entry_point("preprocess")           # 入口：预处理节点
        workflow.add_edge("preprocess", "agent")         # 预处理 → agent
        workflow.add_conditional_edges("agent", self._router, {  # agent → router 分支
            "tools": "tools",  # _router 返回 "tools" → 跳到 tools 节点
            END: END,           # _router 返回 END → 图结束
        })
        workflow.add_edge("tools", "agent")              # 工具执行完 → 回到 agent

        # 初始化 checkpoint 持久化存储
        # 优先使用 SQLite（生产环境），失败回退到内存（开发/测试）
        db_dir = Path.cwd() / ".claude-code-agent"       # 存储目录
        db_dir.mkdir(parents=True, exist_ok=True)         # 确保目录存在
        db_path = str(db_dir / "checkpoints.db")          # SQLite 数据库路径
        if AsyncSqliteSaver is not None:     # 有 SQLite 依赖 → 尝试 SQLite
            try:
                async def _enter_saver():      # 异步上下文管理器入口
                    ctx = AsyncSqliteSaver.from_conn_string(db_path)  # 创建 SQLite 连接
                    saver = await ctx.__aenter__()                    # 进入异步上下文
                    saver._exit_ctx = ctx      # 保存上下文引用，供 close() 清理
                    return saver
                # 之创建sqlite，不跟其他协程交互，用临时 event loop，
                loop = asyncio.new_event_loop()
                try:
                    checkpointer = loop.run_until_complete(_enter_saver())
                finally:
                    loop.close()               # 立即关闭临时 loop
            except Exception:                  # SQLite 初始化失败（如权限问题）
                checkpointer = MemorySaver()   # 回退到内存
        else:
            checkpointer = MemorySaver()       # 无 SQLite 依赖 → 直接用内存
        self._checkpointer = checkpointer       # 保存引用，供 close() 清理
        return workflow.compile(checkpointer=checkpointer)  # 编译并返回可执行图

    def _router(self, state: AgentStateDict) -> str:
        """条件路由：最后一条消息包含 tool_calls 则路由到 tools 节点，否则 END。
            END：Ctrl+C、
        """
        if self._aborted:           # Ctrl+C 中止 → 直接结束
            return END
        messages = state.get("messages", [])
        if not messages:            # 空消息列表（不应发生，但兜底）
            return END
        last_message = messages[-1] # 取最后一条消息
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"          # LLM 要调工具 → 进入 tools 节点
        # 对话结束（LLM 返回纯文本），打印费用统计
        if not self.is_sub_agent:
            utilization = self.last_llm_call_tokens / self.effective_window if self.effective_window else 0
            print_cost(
                self.total_input_tokens, self.total_output_tokens,
                self.total_cache_read_tokens, self.total_cache_creation_tokens,
                model=self.model_name,
                utilization=utilization,
            )
        return END                   # 对话结束

    # ═════════════════════════════════════════════════════════
    # 图节点实现
    # ═════════════════════════════════════════════════════════

    async def _preprocess_node(self, state: AgentStateDict) -> dict:
        """记忆预取、上下文压缩"""
        messages = state.get("messages", [])

        last_msg = messages[-1] if messages else None
        if isinstance(last_msg, HumanMessage) and not self.is_sub_agent:
            self._memory_query = str(last_msg.content)
            self._memory_message_id = last_msg.id
            await self._refresh_memory_context()

        # 运行压缩流水线（按利用率截断/替换旧工具结果）——针对ToolMessage和messages
        self._run_compression_pipeline(messages)
        # 检查是否需要在轮次边界自动压缩（利用率 > 85%）——针对messages
        await self._check_and_compact(messages)

        # 没触发压缩，则消息列表头部注入 system prompt
        if not any(isinstance(m, SystemMessage) for m in messages):
            plan_suffix = build_plan_mode_prompt(self.plan_file_path) if self.permission_mode == "plan" else ""
            system_text = self._base_system_prompt + plan_suffix
            messages.insert(0, SystemMessage(content=system_text))

        # 返回更新后的消息列表和状态字段，由 LangGraph 自动合并到 state
        return {
            "messages": messages,
        }

    async def _agent_node(self, state: AgentStateDict) -> dict:
        """Agent 节点：调用 LLM 生成回复。
         具体步骤：1.主agent：加载思考动画 2.调用llm流式输出chunk 3.收集chunk返回AIMessage 4.token与llmapi调用统计
                2.子agent：chunk输入缓冲区
        """

        #用户在 tools 节点执行期间按 Ctrl+C，tools 的 while 循环退出，但 tools 节点仍会正常 return。图引擎沿着 tools → agent 边继续，此时如果没有 agent 顶部的，agent 会再发起一次 LLM 调用
        if self._aborted:
            return {}

        # 主 agent 显示加载动画
        if not self.is_sub_agent:
            start_spinner()

        try:
            #调用llm实时流式输出(主)并返回AIMessage 或 放缓存(子)
            response = await self._call_llm(state["messages"])
        finally:
            if not self.is_sub_agent:
                stop_spinner()

        self.last_llm_call_time = time.time()#为压缩做准备

        # ── Token 统计：优先读取 Anthropic 格式（usage_metadata）──
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            # Anthropic 缓存 token 拆分
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_create = usage.get("cache_creation_input_tokens", 0)
            if cache_read or cache_create:
                self.total_cache_read_tokens += cache_read
                self.total_cache_creation_tokens += cache_create
            self.last_llm_call_tokens = input_tokens + output_tokens
        # ── 回退到 OpenAI 格式（response_metadata）──
        elif hasattr(response, "response_metadata"):
            meta = response.response_metadata
            usage = meta.get("usage", {})
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
                # OpenAI: prompt_tokens 含缓存命中，需减去 cached_tokens
                cached = usage.get("cached_tokens", 0) or usage.get("cache_read_input_tokens", 0)
                if cached:
                    prompt_tokens = max(0, prompt_tokens - cached)  # 防止负数
                    self.total_cache_read_tokens += cached
                self.total_input_tokens += prompt_tokens
                self.total_output_tokens += completion_tokens
                self.last_llm_call_tokens = prompt_tokens + completion_tokens

        return {
            "messages": [response],# 列表通常只会有一条AIMessage
            "last_llm_call_time": self.last_llm_call_time,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "last_llm_call_tokens": self.last_llm_call_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "total_cache_creation_tokens": self.total_cache_creation_tokens,
            "output_buffer": self._output_buffer,
        }

    async def _tools_node(self, state: AgentStateDict) -> dict:
        """工具节点：执行最后一条 AIMessage 中的工具调用。

        核心流程:
        1. 提取所有 tool_calls
        2. 预算检查（成本/轮数超限则返回错误信息）
        3. 将并发安全的工具分组，串行校验权限后并发执行
        4. 非并发安全的工具串行执行
        5. 返回 ToolMessage 列表

        并发安全工具（如 read_file、grep_search）在权限预检后通过 asyncio.gather 并发执行。
        权限确认弹窗始终串行处理，避免多个弹窗同时出现。
        """
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}

        self.current_turns += 1  # 每进入一次 tools 节点，轮数加一，用于子agent判断是否exceeded
        budget = self._check_budget()  # 检查是否超预算（成本 / 轮数）
        if budget["exceeded"]:
            print_info(f"Budget exceeded: {budget['reason']}")
            return {"messages": [
                ToolMessage(
                    content=f"Tool call not executed: {budget['reason']}",
                    tool_call_id=tc["id"],
                )
                for tc in last_message.tool_calls
            ]}

        tool_messages: list[ToolMessage] = []  # 收集所有工具执行结果

        # ── 分组执行：并发安全工具批量处理，其余串行 ──
        remaining = list(last_message.tool_calls)  # 待处理的工具调用队列（逐个弹出）

        while remaining and not self._aborted:
            # 阶段 1：收集连续的安全工具，构建并发列表（仅有一个安全工具也会进batch）
            batch = [] # 里面存每一个toolcall{name:..,args..,id=..}
            while remaining and self._can_parallelize(remaining[0]):
                batch.append(remaining.pop(0))
            if batch:#2.执行并发工具调用，压缩调用结果，保存mtime，并构建ToolMessage
                # ── 并发安全工具硬编码直接执行（只读，无需权限检查）──
                for tc in batch:
                    print_tool_call(tc.get("name", ""), tc.get("args", {}))
                #内部的并行工具放在线程池，被os调度，与事件循环同时执行
                results = await asyncio.gather(
                    *[self._execute_one_tool_call_no_permission(tc) for tc in batch],
                    return_exceptions=True,  # 单个失败不影响其他
                )
                #构造ToolMessage(content=xx,tool_call_id=xx)
                for i, result in enumerate(results):
                    tc = batch[i]
                    if isinstance(result, Exception):
                        tool_messages.append(ToolMessage(
                            content=f"Error: {result}", tool_call_id=tc["id"],
                        ))
                    else:
                        tool_messages.append(ToolMessage(
                            content=result, tool_call_id=tc["id"],
                        ))

            # 阶段 2：处理下一个非安全工具（逐个串行执行，含权限确认）
            if remaining:
                tc = remaining.pop(0)
                result_text = await self._execute_one_tool_call(tc)
                msg = ToolMessage(content=result_text, tool_call_id=tc["id"])
                tool_messages.append(msg)

        return {
            "messages": tool_messages, #若有多个工具调用，"messages"列表会有多个ToolMessage
            "current_turns": self.current_turns
        }

    # ═════════════════════════════════════════════════════════
    # 工具辅助
    # ═════════════════════════════════════════════════════════

    @staticmethod
    def _can_parallelize(tc: dict) -> bool:
        """判断工具是否可以与其他安全工具并发执行。"""
        return tc.get("name", "") in CONCURRENCY_SAFE_TOOLS

    async def _execute_one_tool_call_no_permission(self, tc: dict) -> str:
        """线程执行并发工具（无权限校验，无确认弹窗）。"""
        name = tc.get("name", "")
        args = tc.get("args", {})

        raw = await execute_tool(name, args, self._read_file_state)
        result = self._persist_large_result(name, raw)  # 大结果持久化
        print_tool_result(name, result)
        return result

    def _persist_large_result(self, tool_name: str, result: str) -> str:
        """持久化大型工具结果（>30KB），两层压缩，这样避免大型结果撑爆上下文窗口。。

        超过 30KB 的结果不直接传给 LLM，而是:
        1. 写入 .claude-code-agent/tool-results/ 目录
        2. 返回文件路径 + 前 200 行预览（第一层）
        3.对前200行内容再次压缩到50000以内，保留首位各一半
        4. 提示 LLM 使用 read_file 查看完整结果
        """
        THRESHOLD = 30 * 1024  # 30KB 阈值，超过则写文件而非直接回传
        if len(result.encode()) <= THRESHOLD:
            return result
        d = Path.cwd() / ".claude-code-agent" / "tool-results"
        d.mkdir(parents=True, exist_ok=True)
        filename = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}-{tool_name}.txt"
        filepath = d / filename
        filepath.write_text(result, encoding="utf-8")  # 写入文件避免撑爆上下文
        lines = result.split("\n")
        preview = "\n".join(lines[:200])  # 前 200 行预览
        size_kb = len(result.encode()) / 1024 #encode为把字符串（str）转成字节串（bytes）
        return truncate_result(
            f"[Result too large ({size_kb:.1f} KB, {len(lines)} lines). "
            f"Full output saved to {filepath}. Use read_file to see the full result.]\n\n"
            f"Preview (first 200 lines):\n{preview}"
        )

    async def _execute_one_tool_call(self, tc: dict) -> str:
        """端到端执行单个工具调用：打印、执行、打印结果。返回 result_text。"""
        name = tc.get("name", "")
        args = tc.get("args", {})

        print_tool_call(name, args)

        # ── 路由：plan 模式切换工具跳过权限检查，其余（agent/mcp/常规）统一走权限预检 ──
        # 注意：enter/exit_plan_mode 不能过 check_permission，否则 plan 模式下 exit 会被 deny 导致死锁
        if name in ("enter_plan_mode", "exit_plan_mode"):#plan模式含interrupt，和图状态有关，只能放graph.py
            result = await self._execute_plan_mode_tool(name)
        else:
            # 统一权限入口：权限校验 → 分发执行（agent / mcp / 常规）
            result = await self._execute_with_permission(name, args)

        print_tool_result(name, result)
        return result

    #---------------plan模式工具-----------------
    async def _execute_plan_mode_tool(self, name: str) -> str:
        """处理 plan 模式的进入/退出。

        enter_plan_mode: 保存当前权限模式 → 切换到 plan → 生成计划文件路径
        exit_plan_mode:  读取计划文件 → interrupt() 展示给用户审批 → 根据选择返回继续修改或是计划文件内容给llm
        """
        if name == "enter_plan_mode":#当llm判断任务复杂，主动调用这个工具把自己切进 plan 模式。
            if self.permission_mode == "plan":
                return "Already in plan mode."
            self.pre_plan_mode = self.permission_mode  # 保存当前权限模式，退出时恢复
            self.permission_mode = "plan"  # 切换到 plan 模式
            self.plan_file_path = self.generate_plan_file_path()
            print_info(f"Entered plan mode. Plan file: {self.plan_file_path}")
            return f"Entered plan mode. Plan file: {self.plan_file_path}\nWrite your plan incrementally to this file using write_file or edit_file."

        if name == "exit_plan_mode":
            if self.permission_mode != "plan":
                return "Not in plan mode."

            # 从文件读取 LLM 编写的计划内容
            plan_content = "(No plan written)"
            if self.plan_file_path and Path(self.plan_file_path).exists():
                plan_content = Path(self.plan_file_path).read_text(encoding="utf-8", errors="replace")

            if plan_content == "(No plan written)":
                return "No plan file found. Please write a plan first using write_file or edit_file."

            # HITL：暂停图执行，展示计划给用户审批
            # --plan 启动时 pre_plan_mode 为 None，审批时询问执行模式；/plan 或工具进入时 pre_plan_mode 有值，默认恢复之前模式
            #interrupt()后
            #           存：把当前所有状态（对话、token、跑到哪）存起来 = 存档。
            #           停：暂停，把控制权还给外面（问你 y/n）。
            #           恢复：你选了 y/n 后，continue_with通过command+config恢复图状态，interrupt() 返回你的选择。
            answer = interrupt({
                "type": "plan_confirm",
                "plan_content": plan_content,
            })
            confirmed = (answer or {}).get("confirmed", False) if isinstance(answer, dict) else False

            # 用户拒绝 → 继续留在 plan 模式修改计划
            if not confirmed:
                feedback = (answer or {}).get("feedback") if isinstance(answer, dict) and answer else None
                return f"User rejected the plan.\n\nFeedback: {feedback or 'Please revise the plan.'}\n\nRevise your plan and call exit_plan_mode again."

            # 用户同意 → 确定执行模式、清空self.pre_plan_mode和self.plan_file_path
            chosen_mode = (answer or {}).get("mode") if isinstance(answer, dict) else None
            target_mode = chosen_mode or self.pre_plan_mode or "default"
            self.permission_mode = target_mode  # 恢复原权限模式
            self.pre_plan_mode = None
            self.plan_file_path = None
            print_info(f"Plan approved. Executing in {target_mode} mode.")
            return f"User approved the plan.\n\n## Approved Plan:\n{plan_content}\n\nProceed with implementation."

        return f"Unknown plan mode tool: {name}"

    def generate_plan_file_path(self) -> str:
        """生成 plan 模式下的计划文件路径。"""
        d = Path.cwd() / ".claude-code-agent" / "plans"
        d.mkdir(parents=True, exist_ok=True)
        return str(d / f"{self.session_id}.md")

    #---------------权限入口 + 执行分派----------------------
    async def _execute_with_permission(self, name: str, args: dict) -> str:
        """统一权限入口（agent / mcp / 常规工具都走这里）。

        流程:
        1. 预检：check_permission 判断 deny/allow/ask
        2. deny → 返回拒绝信息
        3. ask → 弹窗询问用户 → 用户拒绝返回拒绝信息 → 用户同意则执行
        4. allow → 直接执行
        5. 执行结果超过 30KB 则写入文件，返回文件路径和预览
        """

        perm = check_permission(name, args, self.permission_mode, self.plan_file_path)
        # deny → 直接拒绝，返回原因
        if perm["action"] == "deny":
            print_info(f"Denied: {perm.get('message', '')}")
            return f"Action denied: {perm.get('message', '')}"
        # ask → 弹出 y/n 确认窗口
        if perm["action"] == "ask" and perm.get("message"):
            confirmed = await self._confirm_dangerous(perm["message"])
            if not confirmed:#用户拒绝此工具调用
                return "User denied this action."

        # allow / 用户确认通过 → 分发执行（agent / mcp / 常规）
        raw = await self._dispatch_execute(name, args)
        return self._persist_large_result(name, raw)  # 大结果写文件

    async def _dispatch_execute(self, name: str, args: dict) -> str:
        """纯执行分派（不再管权限）：agent / mcp / 常规工具。"""
        if name in {"memory_search", "memory_save", "memory_delete"}:
            return await self._execute_memory_tool(name, args)
        if name == "agent":
            # 委派搜索任务给只读子 agent（explore）
            return await self._execute_agent_tool(args)
        if self._mcp_manager.is_mcp_tool(name):
            # MCP 协议工具，直接通过 MCP 客户端调用
            return await self._mcp_manager.call_tool(name, args)
        # 常规工具（skill 工具也走这里，返回渲染后的模板）
        raw = await execute_tool(name, args, self._read_file_state)
        return await self._gate2_after_write(name, args, raw)  # Gate 2 写后运行验证

    async def _gate2_after_write(self, name: str, args: dict, raw: str) -> str:
        """Gate 2 写后运行时验证：写 .py 文件成功后自动运行，捕获运行时崩溃与 import 缺失。

        只在 write_file/edit_file 且写入成功（未被 Gate 1 拒绝）时触发；
        非 Python 文件跳过。运行结果（成功/失败/超时）附加到写结果后回传 Agent。
        """
        if name not in ("write_file", "edit_file"):
            return raw
        if raw.startswith(("Error", "Write rejected", "Edit rejected")):
            return raw  # 写失败（含 Gate 1 拒绝）→ 不运行
        file_path = args.get("file_path", "")
        if not file_path.endswith(".py"):
            return raw  # 非 Python 文件 → 不自动运行
        run_result = await asyncio.to_thread(self._auto_run_python, file_path)
        return raw + run_result if run_result else raw

    @staticmethod
    def _auto_run_python(file_path: str) -> str:
        """用当前 Python 解释器运行 .py 文件，捕获运行时错误。返回附加到写结果后的说明文本。"""
        try:
            result = subprocess.run(
                [sys.executable, file_path],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                return f"\n\n[Gate 2 运行验证] 运行成功\nStdout: {result.stdout}"
            return (f"\n\n[Gate 2 运行验证] 运行失败 (exit {result.returncode})\n"
                    f"Stderr: {result.stderr}\nStdout: {result.stdout}")
        except subprocess.TimeoutExpired:
            return "\n\n[Gate 2 运行验证] 运行超时（可能存在死循环）"
        except Exception as e:
            return f"\n\n[Gate 2 运行验证] 运行异常: {e}"

    async def _confirm_dangerous(self, command: str) -> bool:
        """请求用户确认危险操作。

        优先使用外部 confirm_fn 回调（非 CLI 模式），
        否则阻塞等待终端输入 y/n。
        """
        print_confirmation(command)
        if self.confirm_fn:
            return await self.confirm_fn(command)
        try:
            answer = input("  Allow? (y/n): ").strip().lower()
            return answer.startswith("y")
        except (EOFError, KeyboardInterrupt):
            return False

    #--------------agent工具--------------------

    async def _execute_agent_tool(self, args: dict) -> str:
        """处理 agent 工具——委派搜索任务给只读子 Agent（explore）。

        子 Agent 是只读的（read_file/list_files/grep_search），在独立上下文里
        搜索海量文件，跑完只回结论，保护主 Agent 上下文不被大量搜索结果污染。

        子 Agent 配置:
        - 继承主 Agent 的 model
        - 使用 subagent.py 返回的 explore system prompt + 只读工具集
        - permission_mode 固定为 bypassPermissions
        """
        prompt = args.get("prompt") or ""

        config = get_sub_agent_config()  # 取 explore 的 system prompt + 只读工具集
        print_sub_agent_start("explore")
        sub_agent = CodingAgent(  # 创建独立子 Agent 实例
            modelname=self.model_name,  # 继承主 Agent 的模型
            custom_system_prompt=config["system_prompt"],  # explore 的专属 prompt
            custom_tools=config["tools"],  # 只读工具集
            is_sub_agent=True,
            permission_mode="bypassPermissions",  # 只读工具集，无需继承主 agent 权限模式
            max_turns=30,  # 子 agent 最多 30 轮工具调用，防止 explore 无限跑飞
        )
        try:
            sub_result = await sub_agent._run_once(prompt)  # 子 Agent 单次执行
            # 将子 Agent 的 token 消耗合并到主 Agent 统计中
            self.total_input_tokens += sub_result["tokens"]["input"]
            self.total_output_tokens += sub_result["tokens"]["output"]
            print_sub_agent_end("explore")
            # 大结果兜底统一由 _execute_with_permission 外层做，这里返回原始文本
            return sub_result["text"] or "(Sub-agent produced no output)"
        except Exception as e:
            print_sub_agent_end("explore")
            return f"Sub-agent error: {e}"

    async def _run_once(self, prompt: str) -> dict:
        """子 Agent 入口：运行并捕获输出。

        设置 output_buffer 收集所有流式输出，调用 chat() 执行，
        返回包含文本和 token 统计的字典。

        Returns:
            {"text": str, "tokens": {"input": int, "output": int}}
        """
        self._output_buffer = [] #← ① 开启子 agent 模式
        prev_in = self.total_input_tokens
        prev_out = self.total_output_tokens
        await self.chat(prompt) #← ② chat期间流式输出全被收集到 buffer
        text = "".join(self._output_buffer) #③ 拿到结果
        self._output_buffer = None # ← ④ 复位，回到主 agent 模式
        return {
            "text": text,
            "tokens": {
                "input": self.total_input_tokens - prev_in,
                "output": self.total_output_tokens - prev_out,
            },
        }

    # ═════════════════════════════════════════════════════════
    # LLM 创建与调用
    # ═════════════════════════════════════════════════════════

    def _create_llm(self, api_base: str | None, anthropic_base_url: str | None, api_key: str | None):
        """根据配置创建 LangChain chat model。

        优先级:
        1. api_base 或 OPENAI_API_KEY 环境变量存在 → ChatOpenAI
        2. 否则 → ChatAnthropic

        支持 SDK 级别重试（通过 MAX_RETRIES 环境变量设置）。
        thinking 模式仅对 Anthropic 后端生效。
        """
        _sdk_retries: dict[str, Any] = {}
        _rv = os.environ.get("MAX_RETRIES")  # 从环境变量读取 SDK 级别重试次数
        if _rv is not None and _rv != "":
            try:
                _sdk_retries["max_retries"] = int(_rv)
            except ValueError:
                pass

        # 每个模型的最大输出 token 数（用于限制 LLM 生成长度）
        max_tok = self._MAX_OUTPUT_TOKENS.get(self.model_name, 2000)

        # ── OpenAI 兼容后端（DeepSeek、GPT 等）──
        if api_base or os.environ.get("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            resolved_base = api_base or os.environ.get("OPENAI_BASE_URL")  # 优先用传入的 base_url
            kwargs: dict[str, Any] = {"model": self.model_name, "max_tokens": max_tok,"api_key": api_key,"base_url": resolved_base}
            kwargs.update(_sdk_retries)
            return ChatOpenAI(**kwargs)

        # ── Anthropic 原生后端 ──
        from langchain_anthropic import ChatAnthropic
        kwargs: dict[str, Any] = {"model": self.model_name, "max_tokens": max_tok}
        if api_key:
            kwargs["api_key"] = api_key
        if anthropic_base_url:
            kwargs["base_url"] = anthropic_base_url  # 自定义 Anthropic API 地址
        # 扩展思考配置（仅 Anthropic 后端支持）
        if self.thinking:
            if self.model_name in self._ADAPTIVE_THINKING_MODELS:
                # Claude 4.x 支持自适应思考：budget 设为 max_tok - 1
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": max_tok - 1}
            else:
                # 旧模型使用固定 budget（最多 4000 token）
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": min(4000, max_tok - 1)}
        kwargs.update(_sdk_retries)
        return ChatAnthropic(**kwargs)

    async def _call_llm(self, messages: list) -> AIMessage:
        """调用 LLM，支持重试，主agent实时文本流式输出，返回AIMessage；子agent则将文本放缓存

        Anthropic 后端：自动为 system message 和最后一条消息添加 cache_control 断点，
        启用前缀缓存以降低延迟和费用。

        流式输出：实时将 LLM 文本输出到终端（主 agent）或 output_buffer（子 agent）。

        AIMessage(
            content="",
            tool_calls=[
                {"name": "memory_search", "args": {...}, "id": "call_1"},
                {"name": "memory_search", "args": {...}, "id": "call_2"},
        """
        if self._memory_dirty:
            await self._refresh_memory_context()
        messages = self._messages_with_memory(messages)

        # 工具绑定：把工具定义"挂"到 LLM 上，让 LLM 知道自己有哪些工具可用
        if self.tools:
            llm_with_tools = self._llm.bind_tools(  # bind_tools 返回新的 Runnable
                [self._tool_to_openai_format(t) for t in self.tools]
            )
        else:
            llm_with_tools = self._llm

        # Anthropic: 添加 cache_control 断点以启用前缀缓存
        msgs = self._add_cache_breakpoints(messages) if self._is_anthropic else messages

        # 重置流式输出状态（每次 LLM 调用重新开始）
        self._stream_text_started = False  # stop_spinner() 之后为 True，代表已换行，后续 chunk 不再重复换行
        self._streaming_emitted = False   # 流式输出了文字则为True，若流式结束后没输出过则走兜底一次性打印

        # 1.实时流式输出(主agent)或存缓冲区(子agent)
        # 2.收集chunk合并为AIMessage返回，若空流，则chunks为空
        async def _do():
            chunks = []
            # chunk是AIMessageChunk
            async for chunk in llm_with_tools.astream(msgs):  # 异步迭代流式 chunk，每一个chunk是AIMessageChunk
                chunks.append(chunk)  # 累积所有 chunk 供后续合并
                self._emit_streaming_chunk(chunk)# 实时输出文本（主 agent 打印到终端，子 agent 写入 buffer）
            if not chunks: # 空流 -- 回退到非流式调用（兼容某些后端）
               return await llm_with_tools.ainvoke(msgs)
            # 用 reduce 将所有 chunk 合并成完整的 AIMessage
            return reduce(lambda a, b: a + b, chunks)

        # 带重试的 LLM 调用（指数退避 + 中止检查）
        response = await _with_retry(_do, abort_check=lambda: self._aborted)

        # 空流回退（ainvoke 兜底）时，流式没产出过文本 → 一次性输出/收集完整 content
        if response.content and not self._streaming_emitted:
            if self.is_sub_agent:
                self._output_buffer.append(self._extract_chunk_text(response))
            else:
                stop_spinner()                      # 关动画（正常流式路径没执行过，spinner 可能还转着）
                if isinstance(response.content, str):
                    print_assistant_text("\n" + response.content)  # 纯文本响应
                elif isinstance(response.content, list):
                    for block in response.content:  # content blocks 格式（Anthropic）
                        if isinstance(block, dict) and block.get("type") == "text":
                            print_assistant_text(block["text"])
        if self._is_anthropic:
            # 过滤 thinking 内容块：Anthropic 不允许在后续请求中回传 thinking 块
            response = self._filter_thinking_from_message(response)

        return response

    @staticmethod
    def _tool_to_openai_format(t: dict) -> dict:
        """将内部工具 schema 转换为 OpenAI function 格式，供 LangChain 绑定使用。"""
        return {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }

    # ═════════════════════════════════════════════════════════════════════
    # 流式输出：若为主agent，则停止思考动画，加入换行，把chunk打印到终端
    #          若为子agent，则添加到缓冲区
    # ═════════════════════════════════════════════════════════════════════

    def _emit_streaming_chunk(self, chunk) -> None:
        """实时输出流式文本 chunk。

        处理逻辑:
        - 子 agent: 输出到 output_buffer（不直接打印）
        - 主 agent: 直接打印到终端，首次输出前停止 spinner
        """
        text = self._extract_chunk_text(chunk)
        if not text:
            return

        if self.is_sub_agent:
            # 子 agent 模式：收集到内存
            self._output_buffer.append(text)
            self._streaming_emitted = True  # 已有文本产出，避免空流兜底重复收集
        else:
            # 主 agent 模式：实时终端输出
            if not self._stream_text_started:# 如果是第一个chunk，则在动画后加换行
                stop_spinner()  # 首次文本输出前停止加载动画
                sys.stdout.write("\n")#write默认进缓冲区
                sys.stdout.flush() #flush之后才会进终端
                self._stream_text_started = True
            sys.stdout.write(text)
            sys.stdout.flush()
            self._streaming_emitted = True

    @staticmethod  # 告诉读者这个函数不依赖实例状态
    def _extract_chunk_text(chunk) -> str:
        """从流式 chunk 中提取可显示文本。

        兼容两种格式:
        - 字符串 content（大多数模型）
        - 列表 content（Anthropic content blocks 格式）
        """
        content = chunk.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return ""



    # ═════════════════════════════════════════════════════════
    # 预算控制
    # ═════════════════════════════════════════════════════════

    def _check_budget(self) -> dict:
        """检查是否超出成本或轮数预算（在每次工具执行前调用）。

        Returns:
            {"exceeded": bool, "reason": str}
        """
        # 先检查费用上限（美元）
        if self.max_cost_usd is not None and self._get_current_cost_usd() >= self.max_cost_usd:
            return {"exceeded": True, "reason": f"Cost limit reached (${self._get_current_cost_usd():.4f} >= ${self.max_cost_usd})"}
        # 再检查工具调用轮数上限
        if self.max_turns is not None and self.current_turns >= self.max_turns:
            return {"exceeded": True, "reason": f"Turn limit reached ({self.current_turns} >= {self.max_turns})"}
        return {"exceeded": False}

    def _get_current_cost_usd(self) -> float:
        """估算当前累计费用（美元），基于累计 token 数乘以各模型定价。

        Claude 定价（每百万 token）:
        - 输入: $3 / 缓存读取: $0.3 / 缓存写入: $3.75 / 输出: $15

        DeepSeek 定价: ¥1/M 输入, ¥2/M 输出
        """
        M = 1_000_000  # 定价单位：每百万 token
        m = self.model_name.lower()
        if "deepseek" in m:
            return (self.total_input_tokens / M) * 1 + (self.total_output_tokens / M) * 2  # 人民币定价
        return (
            (self.total_input_tokens / M) * 3  # Claude 输入价
            + (self.total_cache_read_tokens / M) * 0.3  # 缓存读取更便宜
            + (self.total_cache_creation_tokens / M) * 3.75  # 缓存写入略贵于普通输入
            + (self.total_output_tokens / M) * 15  # 输出最贵
        )


    # ═════════════════════════════════════════════════════════
    # 记忆系统辅助
    # ═════════════════════════════════════════════════════════

    async def _refresh_memory_context(self) -> None:
        self._memory_context = ""
        self._memory_dirty = False
        if self._memory_store and self._memory_query:
            try:
                memories = await self._memory_store.search_memories(self._memory_query)
                self._memory_context = format_memories_for_injection(memories)
            except ValueError as error:
                print_info(f"Memory recall unavailable: {error}")
            except Exception:
                print_info("Memory recall unavailable. Check Milvus, embedding configuration and /memory reindex.")

    def _messages_with_memory(self, messages: list) -> list:
        """Prefix the current user message's request copy; keep system and history intact."""
        result = list(messages)
        if not self._memory_context:
            return result
        # Tool results can follow the current user message; locate it by its graph ID.
        for index in range(len(result) - 1, -1, -1):
            message = result[index]
            if isinstance(message, HumanMessage):
                if self._memory_message_id is not None and message.id != self._memory_message_id:
                    continue
                prefix = f"【相关历史记忆】\n{self._memory_context}\n\n【当前用户问题】\n"
                content = message.content
                if isinstance(content, str):
                    content = prefix + content
                else:
                    content = [{"type": "text", "text": prefix}] + list(content)
                result[index] = message.model_copy(update={"content": content})
                break
        return result

    async def _execute_memory_tool(self, name: str, args: dict) -> str:
        if self._memory_store is None:
            return "Error: Memory tools are unavailable to sub-agents."
        try:
            if name == "memory_search":
                results = await self._memory_store.search_memories(
                    args["query"], top_k=args.get("limit", 5), threshold=args.get("threshold", 0.35),
                )
                return json.dumps(results, ensure_ascii=False)
            if name == "memory_save":
                result = await self._memory_store.save_memory(args)
                self._memory_dirty = True
                return json.dumps({"saved": asdict(result)}, ensure_ascii=False)
            await asyncio.to_thread(self._memory_store.delete_memory, args["memory_id"])
            self._memory_dirty = True
            return json.dumps({"deleted": args["memory_id"]})
        except (ValueError, KeyError, TypeError) as error:
            return f"Error: {error}"
        except Exception:
            return "Error: Memory operation failed. Check Milvus and embedding service availability; retry and search to verify the outcome."


    # ═════════════════════════════════════════════════════════
    # 上下文压缩
    # ═════════════════════════════════════════════════════════

    async def _check_and_compact(self, messages: list) -> None:
        """当上下文利用率超过 85% 时触发自动压缩。

        压缩策略: 用 LLM 总结对话历史，保留关键信息后清空消息列表重建。
        """
        if self.last_llm_call_tokens > self.effective_window * 0.85:  # 超过 85% 窗口
            print_info("Context window filling up, compacting conversation...")
            await self._compact_conversation(messages)

    async def _compact_conversation(self, messages: list) -> None:
        """执行对话压缩（LLM 总结 + 重建消息列表）。

        步骤:
        1. 用 LLM 总结除最后一条用户消息外的所有对话作为HumanMessage
        2. 清空消息列表
        3. 重建: SystemMessage + 摘要HumanMessage + 指定AIMessage + 最后一条原始用户消息
        """
        if len(messages) < 5:  # 消息太少不值得压缩
            return
        last_user = messages[-1] if isinstance(messages[-1], HumanMessage) else None
        # 调用 LLM 生成对话摘要（保留关键决策、文件路径和上下文）
        summary_resp = await self._llm.ainvoke([
            SystemMessage(content="Summarize the conversation in a concise paragraph, preserving key decisions, file paths, and context."),
            *messages[:-1],  # 传入除最后一条外的所有消息作为上下文
            HumanMessage(content="Summarize the conversation so far in a concise paragraph."),
        ])
        summary = str(summary_resp.content) if summary_resp.content else "No summary."
        messages.clear()  # 清空原消息列表
        # 重建消息列表：system prompt + plan 指令 + 摘要 + 确认 + 最新用户消息
        system_text = self._base_system_prompt
        if self.permission_mode == "plan":
            plan_suffix = build_plan_mode_prompt(self.plan_file_path)  # plan 模式下保留计划指令prompt
            if plan_suffix:
                system_text += plan_suffix
        messages.append(SystemMessage(content=system_text))
        messages.append(HumanMessage(content=f"[Previous conversation summary]\n{summary}"))
        messages.append(AIMessage(content="Understood. I have the context. How can I continue helping?"))

        if last_user:
            messages.append(last_user)  # 保留最后一条用户消息
        self.last_llm_call_tokens = 0  # 重置 token 计数
        print_info("Conversation compacted.")

    def _run_compression_pipeline(self, messages: list) -> None:
        """三级上下文压缩流水线。

        Tier 1 -- 空闲清理（距上次 LLM 调用 ≥ 5 分钟）:
          旧工具结果可能已无关，只保留近3条，其余标记为 [Old result cleared]。

        Tier 2 -- 利用率裁剪（利用率 ≥ 0.75）:
          上下文紧张，只保留近3条，其余替换为占位符 [Content snipped]（可 read_file 重读）。

        Tier 3 -- 内容截断（利用率 > 50%）:
          对保留的 ToolMessage 只保留首尾，中间用省略标记替代。
        """
        utilization = self.last_llm_call_tokens / self.effective_window if self.effective_window else 0  # 窗口利用率

        # Tier 1: 空闲清理——距上次 LLM 调用 ≥ 5 分钟，旧工具结果可能已无关，只保留近3条
        if self.last_llm_call_time and (time.time() - self.last_llm_call_time) >= MICROCOMPACT_IDLE_S:
            tool_msgs = [(i, msg) for i, msg in enumerate(messages) if isinstance(msg, ToolMessage) and msg.content not in (SNIP_PLACEHOLDER, "[Old result cleared]")]
            clear_count = len(tool_msgs) - KEEP_RECENT_RESULTS
            for i in range(max(0, clear_count)):
                tool_msgs[i][1].content = "[Old result cleared]"

        # Tier 2: 利用率裁剪——利用率 ≥ 0.75，上下文紧张，只保留近3条（提示 LLM 可 read_file 重读）
        if utilization >= 0.75:
            tool_msgs = [(i, msg) for i, msg in enumerate(messages) if isinstance(msg, ToolMessage) and msg.content not in (SNIP_PLACEHOLDER, "[Old result cleared]")]
            if len(tool_msgs) > KEEP_RECENT_RESULTS:
                for i in range(len(tool_msgs) - KEEP_RECENT_RESULTS):
                    tool_msgs[i][1].content = SNIP_PLACEHOLDER

        # Tier 3: 内容截断——利用率 > 0.5，对保留的 ToolMessage 只保留首尾
        if utilization > 0.5:
            budget = 15000 if utilization > 0.7 else 30000  # 每个 ToolMessage 内容的最大字符数。
            for msg in messages:
                if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and len(msg.content) > budget:
                    keep = (budget - 80) // 2  # 首尾各保留一半
                    msg.content = msg.content[:keep] + f"\n\n[... budgeted: {len(msg.content) - keep * 2} chars truncated ...]\n\n" + msg.content[-keep:]

    # ═════════════════════════════════════════════════════════
    # 会话持久化
    # ═════════════════════════════════════════════════════════

    def resume_latest(self) -> int:
        """从磁盘加载最近一次会话到 _conversation_messages，返回加载的消息数。

        对应 _session_save() 的逆操作：__main__.py 在 --resume 时调用，
        把上次持久化的历史消息加载回内存，让下一轮 chat() 自动接续。
        同时恢复 session_id / session_start_time，使后续保存继续写入同一会话文件。
        """
        session_id = get_latest_session_id()
        if not session_id:
            print_info("No previous sessions found.")
            return 0
        session = load_session(session_id)
        if not session:
            print_info("No session found to resume.")
            return 0
        raw_messages = session.get("messages")
        if not raw_messages:
            return 0
        try:
            msgs = _json_to_messages(raw_messages)
            self._conversation_messages = msgs
            self.session_id = session_id  # 沿用原 session_id，避免分叉出新会话文件
            meta = session.get("metadata") or {}
            self.session_start_time = meta.get("startTime", self.session_start_time)  # 保持原始开始时间
            print_info(f"Resumed session: {session_id} ({len(msgs)} messages)")
            return len(msgs)
        except Exception:
            return 0

    def _session_save(self) -> None:
        """保存会话到磁盘（JSON 格式）。"""
        try:
            save_session(self.session_id, {
                "metadata": {
                    "id": self.session_id, "model": self.model_name,
                    "cwd": str(Path.cwd()), "startTime": self.session_start_time,
                },
                "messages": _messages_to_json(self._conversation_messages),
            })
        except Exception as e:
            print(f"[session] save failed: {e}", flush=True)

    # ═════════════════════════════════════════════════════════
    # /goal -- 自主目标循环
    # ═════════════════════════════════════════════════════════

    def show_goal(self) -> None:
        """显示当前 goal 状态。"""
        if not self.active_goal:
            print_info("No active goal.")
            return
        secs = time.time() - self.active_goal["started_at"]
        last = f"\n  last reason: {self.active_goal['last_reason']}" if self.active_goal.get("last_reason") else ""
        print_info(f"◎ /goal active\n  condition: {self.active_goal['condition']}\n  iterations: {self.active_goal['iterations']}\n  elapsed: {secs:.1f}s{last}")

    async def pursue_goal(self, condition: str) -> None:
        """设置并执行 goal 自主循环，用一个独立的 Judge LLM 做裁判，而不是 LLM 自己觉得能回复了就结束。

        循环逻辑:
        1. 设置 active_goal 并发送初始指令
        2. 每轮结束后调用 LLM 评估条件是否满足
        3. 满足 → 退出 / 不满足 → 告诉 Agent 继续 / 不可能 → 退出
        4. 检查预算和最大迭代次数
        """
        # 设置 goal 状态（原 set_goal 逻辑）
        self.active_goal = {"condition": condition, "iterations": 0, "started_at": time.time(), "last_reason": None}
        print_info(f'◎ /goal active -- condition: "{condition}"')
        directive = goal_directive(condition)  # 启动指令模板

        self.goal_stop = False  # 重置停止标志
        try:
            await self.chat(directive)  # 首轮对话（内部 agent ⇄ tools 循环，直到 LLM 自然停止）
            while self.active_goal and not self.goal_stop and not self._aborted:
                # 评估当前条件是否满足（用独立 Judge LLM 判定，而非 Agent 自己说了算）
                verdict = await self._evaluate_goal(self.active_goal["condition"])
                if verdict["ok"]:
                    turns = self.active_goal["iterations"] + 1  # 执行 agent + 评估 agent 合计为 1 轮
                    secs = time.time() - self.active_goal["started_at"]
                    print_info(f"✓ Goal achieved ({turns} turns, {secs:.1f}s): {verdict['reason']}")
                    break
                if verdict.get("impossible"):  # 条件被判定为不可能达成
                    print_info(f"Hooks: condition judged impossible: {verdict['reason']}")
                    break
                # 没有完成目标，继续迭代
                self.active_goal["iterations"] += 1
                self.active_goal["last_reason"] = verdict["reason"]
                print_info(f"Hooks: condition was not met: {verdict['reason']}")
                # 预算检查（费用或轮数超限）
                budget = self._check_budget()
                if budget["exceeded"]:
                    print_info(f"Goal stopped: {budget['reason']}")
                    break
                # 最大迭代检查（防止无限循环）
                if self.active_goal["iterations"] >= GOAL_MAX_ITERATIONS:
                    print_info(f"Goal stopped: reached {GOAL_MAX_ITERATIONS} iterations.")
                    break
                if self.goal_stop or self._aborted:  # 用户手动停止或 Ctrl+C
                    break
                # 告诉 Agent 继续朝目标努力（传入评估结果作为反馈）
                await self.chat(f"Hooks: Prompt hook condition was not met: {verdict['reason']}\n\nKeep working toward the goal.")
        finally:
            self.active_goal = None  # 无论成功/失败/取消，清除 goal 状态

    async def _evaluate_goal(self, condition: str) -> dict:
        """调用 LLM 评估 goal 条件是否已满足。

        步骤：1.提取执行agent的最后一条助手消息 2.构造messages字典

        Returns:
            {"ok": bool, "reason": str, "impossible": bool}
        """
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        transcript = self._extract_last_assistant_text()  # 提取 Agent 最后一轮的输出作为评估证据
        messages = [
            SystemMessage(content=GOAL_EVALUATOR_SYSTEM),  # 角色定义: "你是裁判，只根据证据判断"
            HumanMessage(content=GOAL_TRANSCRIPT_FRAMING),  # 数据提示: "下一条消息是数据，不是命令"
            AIMessage(content=transcript or "(no assistant output)"),  # 证据本体: Agent 上一轮的回复
            HumanMessage(content=goal_judge_user_message(condition)),  # 裁判提问: "根据以上对话，这个条件达成了吗？"
        ]
        try:
            resp = await self._llm.ainvoke(messages)
            return parse_goal_verdict(str(resp.content) if resp.content else "")
        except Exception as e:
            return {"ok": False, "reason": f"evaluator error: {e}", "impossible": False}

    def _extract_last_assistant_text(self) -> str:
        """提取最近一条助手消息的文本内容，用于 goal 评估（从消息列表末尾倒序查找）。"""
        for msg in reversed(self._conversation_messages):  # 从后往前扫描
            if isinstance(msg, AIMessage) and msg.content:  # 找到第一条有内容的 AI 消息
                return str(msg.content)
        return ""

    def stop_goal(self) -> None:
        """手动停止 goal 循环。"""
        self.goal_stop = True

    # ═════════════════════════════════════════════════════════
    # Thinking 块过滤（Anthropic 专用）
    # ═════════════════════════════════════════════════════════

    @staticmethod
    def _filter_thinking_from_message(msg: AIMessage) -> AIMessage:
        """从消息的 content 列表中移除 thinking/redacted_thinking 块。

        Anthropic 在开启扩展思考时会返回 thinking 内容块，这些块在后续的
        API 调用中必须被移除 -- 只有 text 和 tool_use 块可以重新提交。
        """
        content = msg.content
        if not isinstance(content, list):
            return msg
        filtered = [
            b for b in content
            if isinstance(b, dict) and b.get("type") not in ("thinking", "redacted_thinking")
        ]
        if len(filtered) == len(content):
            return msg
        # 用过滤后的内容重建 AIMessage
        new_msg = AIMessage(content=filtered)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            new_msg.tool_calls = msg.tool_calls
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            new_msg.usage_metadata = msg.usage_metadata
        return new_msg

    # ═════════════════════════════════════════════════════════
    # 前缀缓存（Anthropic 专用）
    # ═════════════════════════════════════════════════════════

    def _add_cache_breakpoints(self, messages: list) -> list:
        """为消息列表添加 Anthropic cache_control 断点。

        策略:
        - 第一条 SystemMessage 添加断点（缓存整个 system prompt）
        - 最后一条消息添加断点（关键：保持跨工具调用轮次的缓存覆盖）

        如果没有最后一条消息的断点，第一次工具执行后缓存命中率会骤降，
        因为消息列表以 ToolMessage 结尾而非 Human/AI 消息。

        返回新列表，不修改原始消息。
        """
        result = list(messages)

        # 第一条 SystemMessage 添加缓存断点（缓存整个 system prompt，后续请求命中）
        for i, msg in enumerate(result):
            if isinstance(msg, SystemMessage) and isinstance(msg.content, str):
                result[i] = SystemMessage(content=[
                    {"type": "text", "text": msg.content,
                     "cache_control": {"type": "ephemeral"}}  # ephemeral: 5分钟 TTL
                ])
                break

        # 最后一条消息添加缓存断点 -- 按消息类型分别处理
        if result:
            last = result[-1]
            if isinstance(last, HumanMessage) and isinstance(last.content, str):
                # HumanMessage: 将纯文本包装为 content block 并加断点
                result[-1] = HumanMessage(content=[
                    {"type": "text", "text": last.content,
                     "cache_control": {"type": "ephemeral"}}
                ])
            elif isinstance(last, AIMessage) and isinstance(last.content, str) and last.content:
                # AIMessage: 保留 tool_calls 的同时加 content block 断点
                result[-1] = AIMessage(
                    content=[
                        {"type": "text", "text": str(last.content),
                         "cache_control": {"type": "ephemeral"}}
                    ],
                    tool_calls=last.tool_calls if hasattr(last, 'tool_calls') else None,
                )
            elif isinstance(last, ToolMessage) and isinstance(last.content, str):
                # ToolMessage: 使用 tool_result 类型包裹，在工具结果轮次中保持缓存
                result[-1] = ToolMessage(
                    content=[
                        {"type": "tool_result",
                         "tool_use_id": last.tool_call_id,  # 关联原始 tool_use
                         "content": last.content,
                         "cache_control": {"type": "ephemeral"}}
                    ],
                    tool_call_id=last.tool_call_id,
                )

        return result

# ═══════════════════════════════════════════════════════════════
# 会话序列化工具函数
# ═══════════════════════════════════════════════════════════════

def _messages_to_json(messages: list[BaseMessage]) -> list[dict]:
    """将 LangChain 消息列表转换为 JSON 可序列化的字典列表。

    类型缩写:
    - s: SystemMessage
    - h: HumanMessage
    - a: AIMessage（含可选的 tool_calls）
    - tl: ToolMessage（含 tool_call_id）
    - ?: 未知类型（兜底）
    """
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"t": "s", "c": msg.content})  # s = SystemMessage
        elif isinstance(msg, HumanMessage):
            result.append({"t": "h", "c": msg.content})  # h = HumanMessage
        elif isinstance(msg, AIMessage):
            entry: dict = {"t": "a", "c": msg.content}  # a = AIMessage
            if msg.tool_calls:
                entry["tc"] = msg.tool_calls  # tc = tool_calls（可选）
            result.append(entry)
        elif isinstance(msg, ToolMessage):
            result.append({"t": "tl", "c": msg.content, "id": msg.tool_call_id})  # tl = TooL, id 关联 tool_use
        else:
            result.append({"t": "?", "c": str(msg.content)})  # ? = 未知类型兜底
    return result


def _json_to_messages(data: list[dict]) -> list[BaseMessage]:
    """从序列化的字典列表重建 LangChain 消息对象。

    这是 _messages_to_json 的逆操作，用于会话恢复。
    """
    messages = []
    for d in data:
        t = d.get("t", "?")  # 获取类型标记
        if t == "s":
            messages.append(SystemMessage(content=d["c"]))  # s → SystemMessage
        elif t == "h":
            messages.append(HumanMessage(content=d["c"]))  # h → HumanMessage
        elif t == "a":
            msg = AIMessage(content=d.get("c", ""))  # a → AIMessage
            if "tc" in d:
                msg.tool_calls = d["tc"]  # 恢复 tool_calls
            messages.append(msg)
        elif t == "tl":
            messages.append(ToolMessage(content=d["c"], tool_call_id=d.get("id", "")))  # tl → ToolMessage
        else:
            messages.append(HumanMessage(content=str(d.get("c", ""))))  # ? → 回退为 HumanMessage
    return messages
