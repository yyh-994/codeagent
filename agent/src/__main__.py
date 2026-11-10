"""CLI 入口 — 交互式 REPL + LangGraph Agent。

完整启动流程:
  main()
    ├─ ① 解析命令行参数（--yolo / --plan / --thinking / --resume / --max-cost / --max-turns）
    ├─ ② 根据参数决定 permission_mode（default / bypassPermissions / plan）
    ├─ ③ 从环境变量读取模型配置，创建 CodingAgent 实例
    ├─ ④ 如果是 one-shot 模式（命令行带用户消息），直接调一次 agent.chat() 后退出
    └─ ⑤ 否则进入 _run_repl() 交互循环

REPL 循环 (_run_repl):
  打印欢迎语 → 循环:
    ├─ 读取用户输入
    ├─ 匹配 REPL 命令（/plan /yolo /clear /cost /compact /goal /memory）
    │   每个命令内部直接调 agent 的对应方法（如 agent.pursue_goal）
    └─ 其他 → _safe_chat(msg) → agent.chat(msg)
                                    │
                                    ├─ preprocess（记忆预取 + 压缩 + system prompt）
                                    └─ agent ⇄ tools（LangGraph 内部循环直到 LLM 停）
                                              │
                                    LLM 返回 tool_calls 时 router 走到 tools 节点
                                    tools 节点执行完回到 agent 节点
                                    LLM 返回纯文本时 router → END

中断处理 (_handle_interrupt):
  LangGraph 的 interrupt() 机制让图暂停，等用户确认后再 continue_with() 恢复。
  当前支持两种中断:
    - plan_approval: Plan 模式下展示审批计划，y/n 确认
    - 工具确认: 由 tools.py 中的 confirm() 触发

Ctrl+C 处理 (handle_sigint):
  - 第一次: stop_goal() + abort() → 取消当前运行的 chat()
  - 第二次: 退出程序
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

from .graph import CodingAgent
from .ui import (
    print_welcome, print_user_prompt, print_error, print_info,
    print_plan_for_approval,
    print_confirmation,
)


# ═══════════════════════════════════════════════════════════════
# CLI 参数解析
# ═══════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    """解析命令行参数，返回 Namespace 对象。

    调用位置: main()，启动时最先执行。

    用法示例:
      claude-code-agent                                  # 交互 REPL
      claude-code-agent "fix the bug"                    # one-shot，无 REPL
      claude-code-agent --yolo "refactor"                # 跳过所有确认
      claude-code-agent --plan                           # Plan 模式（只读）
      claude-code-agent --resume                         # 恢复上次会话
      claude-code-agent --max-turns 10 "do something"    # 限制工具调用轮数
      claude-code-agent --max-cost 0.5 "do something"    # 限制费用
    """
    parser = argparse.ArgumentParser(
        prog="claude-code-agent",
        description="Claude Code Agent — LangGraph-powered coding agent",
        add_help=False,  # 自定义 help 输出，不用默认的
    )
    # 位置参数: 如果命令行末尾带了用户消息文本，进入 one-shot 模式
    parser.add_argument("umsg", nargs="*", help="One-shot user message")
    # 权限模式开关
    parser.add_argument("--yolo", action="store_true",
                        help="Skip all confirmation prompts")
    parser.add_argument("--plan", action="store_true",
                        help="Plan mode: read-only")
    # 模型开关
    parser.add_argument("--thinking", action="store_true",
                        help="Enable extended thinking")
    # 恢复开关
    parser.add_argument("--resume", action="store_true",
                        help="Resume last session")
    # 预算限制
    parser.add_argument("--max-cost", type=float, default=None,
                        help="Max USD spend")
    parser.add_argument("--max-turns", type=int, default=100,
                        help="Max agentic turns (default: 100)")
    parser.add_argument("--help", "-h", action="store_true",
                        help="Show help")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# 权限模式映射
# ═══════════════════════════════════════════════════════════════

def _resolve_permission_mode(args: argparse.Namespace) -> str:
    """把命令行参数映射为 Agent 的权限模式字符串。

    调用位置: main()，创建 CodingAgent 前调用。

    三种模式:
      "bypassPermissions" — --yolo 开启，跳过所有工具确认弹窗
      "plan"              — --plan 开启，只读模式，LLM 只生成计划不执行
      "default"           — 默认模式，每次工具调用弹出确认

    注意: --yolo 和 --plan 互斥，--yolo 优先。
    """
    if args.yolo:
        return "bypassPermissions"
    if args.plan:
        return "plan"
    return "default"


# ═══════════════════════════════════════════════════════════════
# REPL 交互循环
# ═══════════════════════════════════════════════════════════════
#
# REPL = Read（读用户输入）→ Eval（判断命令/普通聊天）→ Print（流式输出 LLM 回复）→ Loop（回到 Read）
#
# _run_repl 内有两个嵌套函数:
#   _handle_interrupt() — 处理 LangGraph 的 HITL（Human-In-The-Loop）中断
#   _safe_chat()        — 包装 agent.chat()，捕获异常并触发中断处理
#   handle_sigint()     — Ctrl+C 信号处理

async def _run_repl(agent: CodingAgent) -> None:
    """交互式 REPL 循环，使用 LangGraph 原生的 interrupt() 实现 HITL。

    LangGraph 的 interrupt() 会暂停图的执行，把控制权交还给调用方。
    REPL 拿到中断信息后:
      - plan_approval → 展示计划，让用户 y/n
      - 工具确认 → 展示工具调用，让用户 y/n
    用户选择后调 agent.continue_with() 恢复图执行。
    """

    # ═══════════════════════════════════════════════════════
    # 中断处理（HITL）
    # ═══════════════════════════════════════════════════════

    async def _handle_interrupt() -> None:
        """检查并处理当前中断。

        调用位置:
          - _safe_chat() 中，每次 agent.chat() 返回后调一次
          - plan_approval 的 y 分支，continue_with() 后递归调自己，
            因为图恢复后可能又遇到下一个中断

        中断类型:
          - "plan_approval": Plan 模式下 LLM 生成了计划，让用户审批
            （由 graph.py 的 exit_plan_mode 调用 interrupt() 触发，目前是唯一的中断类型）

        注意: 危险工具的确认不走 interrupt，而是由 graph.py 的
        _confirm_dangerous() 用 input()/confirm_fn 同步阻塞询问。
        """
        if not agent.is_interrupted():
            return  # 没有未处理的中断，直接返回

        pending = agent._interrupt_pending  # interrupt() 携带的数据
        kind = pending.get("type", "") #中断类型，目前只有plan_confirm

        # ── plan模式，只生成计划文件，退出plan后提交审批 ，y则执行，n则继续修改──
        #三种plan模式启动方式：--plan（执行时询问用户以yolo还是default）、/plan（以进入plan之前的模式执行）、工具调用（以进入plan之前的模式执行）。
        if kind == "plan_confirm":
            plan_content = pending.get("plan_content", "")
            print_plan_for_approval(plan_content)
            # 循环等用户输入，直至输入 y或n
            while True:
                try:
                    answer = input(" 执行计划？(y/n): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    if not agent.is_interrupted():  # 中断已被 abort 丢弃 → 返回，不再继续审批
                        return
                    answer = "n"  # 异常视为拒绝
                if answer.startswith("y"):
                    # 用户同意 → 恢复图执行，confirmed=True
                    mode = None
                    if agent.pre_plan_mode is None:
                        # --plan 启动（pre_plan_mode 从未被设置）：询问以什么模式执行
                        try:
                            choice = input("  以什么模式执行？[1] default [2] yolo（回车=default）: ").strip()
                        except (EOFError, KeyboardInterrupt):
                            choice = ""
                        mode = "bypassPermissions" if choice == "2" else "default"
                    await agent.continue_with({"confirmed": True, "mode": mode})
                    break
                elif answer.startswith("n"):
                    # 用户拒绝 → 让用户输入修改意见（可选）
                    try:
                        feedback = input("  修改意见（直接回车跳过）: ").strip()
                    except EOFError:
                        feedback = ""
                    await agent.continue_with({
                        "confirmed": False,
                        "feedback": feedback or None,
                    })
                    break

            # 恢复后检查有没有新的中断（递归处理）
            while agent.is_interrupted():
                await _handle_interrupt()

        # ── 工具确认（不经过 interrupt） ──
        # 危险工具的确认由 graph.py 的 _confirm_dangerous() 用 input()/confirm_fn
        # 同步阻塞完成，不触发 interrupt()，因此这里无需处理。

    # ═══════════════════════════════════════════════════════
    # 安全聊天包装
    # ═══════════════════════════════════════════════════════

    async def _safe_chat(msg: str) -> None:
        """包装 agent.chat():
        1. 调 agent.chat(msg) 执行 LangGraph 图
        2. 图结束后检查是否有未处理的中断
        3. 有中断 → 调 _handle_interrupt() 让用户确认
        4. 确认后图可能继续执行并再次中断 → 循环直到所有中断处理完

        异常处理: abort 相关异常静默吞掉，其他异常打印错误信息。
        """
        try:
            await agent.chat(msg)
            # chat() 返回后，可能图在某个节点调了 interrupt() 还没处理
            while agent.is_interrupted():
                await _handle_interrupt()
        except Exception as e:
            if "abort" not in str(e).lower():
                print_error(str(e))

    # ═══════════════════════════════════════════════════════
    # Ctrl+C 信号处理
    # ═══════════════════════════════════════════════════════
    #
    # 信号在类 Unix 系统中由内核发送；Windows 上 Python 在控制台进程内模拟。
    # signal.signal(SIGINT, handler) 注册后:
    #   - 用户按 Ctrl+C → OS/控制台向本进程发送 SIGINT
    #   - 当前正在执行的 Python 代码被中断，handler 被调用
    #   - handler 执行完毕后，被中断的代码抛 KeyboardInterrupt
    #
    # 两段式处理（不管当前在做什么，语义统一）:
    #   第一次 Ctrl+C — abort() 结束当前 chat（跑图→cancel / 审批等待→丢弃中断 / 空闲→无操作）
    #   第二次 Ctrl+C — 退出程序

    sigint_count = 0  # 连按计数，新输入时清零

    def handle_sigint(sig, frame):
        """Ctrl+C 信号处理器。

        sig:  信号编号（SIGINT = 2）
        frame: 当前调用栈帧（未使用）

        注意: 信号处理器可能在主线程 print() 写到 stdout 的中途被调用，
        在处理器里 print() 会造成 reentrant call 崩溃（RuntimeError）。
        所以这里用 os.write() 直接写 stderr（原子 syscall、无缓冲、不会 reentrant），
        退出用 os._exit()（立即终止，跳过 atexit 清理，避免 reentrant flush）。
        """
        nonlocal sigint_count

        # 先尝试优雅停止 goal 循环
        agent.stop_goal()

        sigint_count += 1
        if sigint_count == 1:
            # 第一次：结束当前 chat（abort 内部统一处理各种状态）
            agent.abort()
            os.write(sys.stderr.fileno(), b"\n  (interrupted) - press Ctrl+C again to exit\n")
        else:
            # 第二次：退出程序
            os.write(sys.stderr.fileno(), b"\nBye!\n")
            os._exit(0)
    #注册处理器函数，用户按CtrlC后先执行handle_sigint，再在刚刚的执行的地方抛KeyboardInterrupt
    signal.signal(signal.SIGINT, handle_sigint)

    # ═══════════════════════════════════════════════════════
    # REPL 主循环
    # ═══════════════════════════════════════════════════════

    print_welcome()

    while True:
        # ── 打印提示符 ──
        print_user_prompt()

        # ── 读一行用户输入 ──
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            # EOFError: stdin 关闭（管道输入结束 / Ctrl+D）
            # KeyboardInterrupt: sigint 处理后信号处理器又没 abort 成功
            print("\nBye!\n")
            break

        inp = line.strip()
        sigint_count = 0  # 新输入 → 重置 Ctrl+C 计数器

        # ── 空行跳过 ──
        if not inp:
            continue

        # ── 退出命令 ──
        if inp in ("exit", "quit"):
            print("\nBye!\n")
            break

        # ── /clear — 清空对话历史 ──
        # 清掉 _conversation_messages 和 LangGraph checkpoint
        if inp == "/clear":
            agent.clear_history()
            continue

        # ── /plan — 乒乓开关：切换 Plan 模式 ──
        # Plan 模式: 只读，LLM 只生成执行计划不写代码
        # 再按一次 /plan 恢复为之前的权限模式
        if inp == "/plan":
            if agent.permission_mode == "plan":
                # 退出 Plan 模式 → 恢复进入前的权限模式
                agent.permission_mode = agent.pre_plan_mode or "default"
                agent.pre_plan_mode = None
                agent.plan_file_path = None
                print_info(f"Exited plan mode → {agent.permission_mode} mode")
            else:
                # 进入 Plan 模式 → 记住当前模式，切到 plan
                agent.pre_plan_mode = agent.permission_mode
                agent.permission_mode = "plan"
                agent.plan_file_path = agent.generate_plan_file_path()
                print_info(f"Entered plan mode. Plan file: {agent.plan_file_path}")
            continue

        # ── /yolo — 乒乓开关：跳过所有确认 ──
        if inp == "/yolo":
            if agent.permission_mode == "bypassPermissions":
                agent.permission_mode = "default"
                print_info("Exited yolo mode → default mode")
            else:
                agent.permission_mode = "bypassPermissions"
                print_info("Entered yolo mode — all confirmations bypassed")
            continue

        # ── /cost — 显示当前 Token 用量和费用 ──
        if inp == "/cost":
            agent.show_cost()
            continue

        # ── /compact — 手动触发对话压缩 ──
        if inp == "/compact":
            try:
                await agent.compact()
            except Exception as e:
                print_error(str(e))
            continue

        # ── /goal — 自主目标循环 ──
        # 调用链:
        #   /goal <条件> → agent.pursue_goal(条件)：设置并执行 goal 循环
        #   pursue_goal 内部: chat → Judge LLM 评估 → 继续/退出
        #   /goal（无参数） → agent.show_goal() 显示当前状态
        if inp == "/goal" or inp.startswith("/goal "):
            condition = inp[len("/goal"):].strip()  # 去掉 "/goal" 前缀，拿到条件文本
            if not condition:
                agent.show_goal()  # 无参数 = 查看状态
                continue
            try:
                await agent.pursue_goal(condition)      # 设置并执行 goal 循环
            except Exception as e:
                if "abort" not in str(e).lower():
                    print_error(str(e))
            continue

        if inp == "/memory reindex":
            try:
                count = await agent.reindex_memories()
                print_info(f"Reindexed {count} memories.")
            except Exception:
                print_error("Memory reindex failed. Check Milvus and embedding service availability.")
            continue

        # ── /memory — 列出所有已保存的记忆 ──
        if inp == "/memory":
            try:
                memories = await asyncio.to_thread(agent.list_memories)
            except Exception:
                print_error("Cannot list memories. Check MILVUS_URI, MILVUS_TOKEN and Milvus availability.")
                continue
            if not memories:
                print_info("No memories saved yet.")
            else:
                print_info(f"{len(memories)} memories:")
                for m in memories:
                    print(f"    {m.id} [{m.type}] {m.content}")
            continue

        # ── 普通聊天: 不属于任何命令 → 发给 agent.chat() ──
        await _safe_chat(inp)

    # REPL 退出 → 清理资源
    await agent.close()


# ═══════════════════════════════════════════════════════════════
# 程序入口
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    """程序入口。

    流程:
      1. 加载 .env 环境变量
      2. 解析命令行参数
      3. 决定权限模式
      4. 从环境变量读取模型配置 → 创建 CodingAgent
      5. 如果有用户消息 → one-shot 模式（调用一次 chat() 退出）
         否则 → 进入 REPL 交互循环
    """

    # ── 加载 .env ──
    # 读取两个位置: agent 源码目录和用户运行 agent 时所在的目录
    from dotenv import load_dotenv
    for env_path in [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]:
        if env_path.exists():
            load_dotenv(env_path)

    # ── 解析启动时的用户输入命令（E：claude-code-agent --yolo，则args. ） ──
    args = _parse_args()

    # ── 显示帮助 ──
    if args.help:
        print("""
        用法: claude-code-agent [选项] [提示词]
        
        选项:
          --yolo, -y          跳过所有确认弹窗（bypassPermissions 模式）
          --plan              计划模式：只读，只描述改动不执行
          --thinking          启用深度思考模式
          --resume            恢复上次会话
          --max-cost USD      预估费用超过指定金额时停止
          --max-turns N       执行 N 轮工具调用后停止
          --help, -h          显示此帮助信息
        
        REPL 命令:
          /clear              清除对话历史
          /plan               切换计划模式（只读）
          /yolo               切换 yolo 模式（跳过所有确认）
          /cost               显示 token 用量和费用
          /compact            手动压缩对话
          /goal <条件>        跨轮次追踪目标
          /memory             列出已保存的记忆
          /memory reindex     更换 embedding 模型后重建所有向量
        
        示例:
          # DeepSeek（最简单）
          set DEEPSEEK_API_KEY=sk-your-key
          claude-code-agent "修复 src/app.py 中的 bug"
          claude-code-agent -d -m deepseek-reasoner "解释这段代码"
        
          # Anthropic
          set ANTHROPIC_API_KEY=sk-ant-xxx
          claude-code-agent "你好"
        
          # OpenAI 兼容
          set OPENAI_API_KEY=sk-xxx
          set OPENAI_BASE_URL=https://api.openai.com/v1
          claude-code-agent --resume
          claude-code-agent  # 启动交互式 REPL
        
        基于 LangGraph 构建 — agent 循环由 StateGraph 驱动。
        """)
        sys.exit(0)

    # ── 决定权限模式 ──
    permission_mode = _resolve_permission_mode(args)

    # ── 创建 Agent ──
    # 模型配置从环境变量读取: MODEL / DEEPSEEK_API_KEY / API_BASE
    model = os.environ.get("MODEL", "deepseek-chat")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print_error("DEEPSEEK_API_KEY is required. Set it in .env or environment.")
        sys.exit(1)
    api_base = os.environ.get("API_BASE", "https://api.deepseek.com/v1")

    #必备，权限模式、最大费用、最大轮次、模型、url、key
    agent = CodingAgent(
        permission_mode=permission_mode,
        modelname=model,
        thinking=args.thinking,
        max_cost_usd=args.max_cost,    # 单次会话累计费用上限（美元）
        max_turns=args.max_turns,      # 单次 chat 内 agent ⇄ tools 最大轮数
        api_base=api_base,
        api_key=api_key,
        resume=args.resume,            # --resume：构造时从磁盘恢复上次会话
    )

    # ── 如果命令行带了用户消息 → one-shot 模式 ──
    umsg = " ".join(args.umsg) if args.umsg else None

    if umsg:
        # One-shot: 调一次 chat() 后退出
        async def _one_shot() -> None:
            try:
                await agent.chat(umsg)
            finally:
                await agent.close()

        try:
            #asyncio.run() — 从同步代码启动整个异步世界，注册为当前主线程的事件循环。事件循环跑协程的过程中，若协程内部由geteventloop，必须注册，即seteventloop。
            #await — 在异步函数内部调用另一个异步函数
            asyncio.run(_one_shot())
        except Exception as e:
            print_error(str(e))
            sys.exit(1)
    else:
        # 无用户消息 → 交互式 REPL
        asyncio.run(_run_repl(agent))


if __name__ == "__main__":
    main()
