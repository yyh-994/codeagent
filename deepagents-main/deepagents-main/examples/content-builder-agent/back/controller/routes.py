"""所有 API 路由 — SSE 实时流 + 认证"""

import asyncio
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, ToolMessage

from back.controller.auth import get_current_user, UserContext
from back.schemas import TaskSubmitRequest

router = APIRouter()

def _friendly_error(e: Exception) -> str:
    """将底层异常映射为用户可读的中文提示。"""
    msg = str(e).lower()
    if "429" in msg or "rate" in msg or "too many" in msg:
        return "当前使用人数较多，请稍等几秒后再试"
    if "401" in msg or "unauthorized" in msg or "auth" in msg:
        return "API 认证失败，请联系管理员检查模型配置"
    if "timeout" in msg or "timed out" in msg:
        return "模型响应超时，请重试或缩短问题"
    if "connection" in msg or "refused" in msg or "unreachable" in msg:
        return "模型服务暂时无法连接，请稍后重试"
    if "token" in msg and ("limit" in msg or "exceed" in msg):
        return "对话内容过长，请缩短后重试"
    return f"服务异常，请稍后重试（{str(e)[:100]}）"


_memory_tasks: set = set()  # 后台记忆提取任务的强引用（asyncio 对 task 只持弱引用，防中途被 GC）


def _sse_once(text: str) -> StreamingResponse:
    """把单条提示文本包成 SSE 流返回（追问/兜底用）。"""
    async def _gen():
        yield f"event: content\ndata: {json.dumps({'content': text}, ensure_ascii=False)}\n\n"
    return StreamingResponse(_gen(), media_type="text/event-stream")


async def _stream_agent_output(agent, input_, *, config, intent, raw_user_msg, request: Request):
    """通用 SSE 流生成器 — 主请求和 resume 共用。"""
    seen_tools = set()
    collected_text = []

    try:
        async for msg, _metadata in agent.astream(
            input_, config=config, stream_mode="messages",
        ):
            # 客户端断开时立即停止 agent
            if await request.is_disconnected():
                logging.getLogger("routes").info("Client disconnected, stopping agent")
                return

            # ── AI 流式增量 (AIMessageChunk) ──
            if isinstance(msg, AIMessageChunk):
                for tc in getattr(msg, "tool_call_chunks", None) or []:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if name and name not in seen_tools:
                        seen_tools.add(name)
                        yield f"event: tool_call\ndata: {json.dumps({'tool_call': name}, ensure_ascii=False)}\n\n"

                # R1 reasoning_content → 推理过程（进思考区）
                reasoning = (
                    getattr(msg, 'reasoning_content', None) or
                    (msg.additional_kwargs or {}).get('reasoning_content', '')
                )
                if reasoning and str(reasoning).strip():
                    yield f"event: thinking\ndata: {json.dumps({'thinking': str(reasoning)}, ensure_ascii=False)}\n\n"

                delta = getattr(msg, "content", "")
                if isinstance(delta, str) and delta.strip():
                    collected_text.append(delta)
                    yield f"event: content\ndata: {json.dumps({'content': delta}, ensure_ascii=False)}\n\n"
                elif isinstance(delta, list):
                    for block in delta:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text.strip():
                                collected_text.append(text)
                                yield f"event: content\ndata: {json.dumps({'content': text}, ensure_ascii=False)}\n\n"

            # ── 工具结果 (ToolMessage) ──
            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "")
                result = str(getattr(msg, "content", ""))
                yield f"event: tool_result\ndata: {json.dumps({'tool_result': {'tool': tool_name, 'content': result[:3000]}}, ensure_ascii=False)}\n\n"

        # ── 语义缓存写入（RAG 问答题结果）──
        if intent == "rag" and collected_text:
            from back.service.rag.semantic_cache import store as semcache_store
            await semcache_store(raw_user_msg, "".join(collected_text))

        # ── HITL 中断检测 ──
        state = agent.get_state(config) #读最新checkpoint快照，取中断信息，比如在哪个工具中断
        if state.interrupts:
            for interrupt in state.interrupts:
                yield f"event: interrupt\ndata: {json.dumps(interrupt, ensure_ascii=False)}\n\n"

    except Exception as e:
        logging.getLogger("routes").exception("SSE stream error")
        friendly = _friendly_error(e)
        yield f"event: error\ndata: {json.dumps({'error': friendly}, ensure_ascii=False)}\n\n"


# ── SSE 实时流（边想边出）──
# 参数校验时，若是BaseModel，默认解析请求体进行校验；若是depends，默认对请求头操作；若是Request，则是对request操作
@router.post("/runs/stream", tags=["任务"])
async def stream_agent(
    body: TaskSubmitRequest,
    request: Request,
    ctx: UserContext = Depends(get_current_user),
):
    """SSE 实时流式 Agent 响应 — 替代 LangGraph 内置 /runs/stream。
    7
    以 Server-Sent Events 推送思考过程：
    - event: thinking     → 模型推理过程（R1 reasoning_content）
    - event: content      → 正式回复文本
    - event: tool_call    → 工具调用通知
    - event: file_content → 文件写入内容（博客正文等）
    - event: image_gen    → 图片生成通知
    """
    from back.service.intent.intent_service import process_task, QuickReply, TaskReply

    uid = ctx.user_id
    role = ctx.role

    messages_data = body.input.get("messages", [])
    config = body.config or {}
    config.setdefault("configurable", {})
    thread_id = config["configurable"].get("thread_id", "default")
    intent_override = config["configurable"].get("intent", "auto")

    # 注入用户身份，供 agent 中间件/tools 通过 config 读取
    config["configurable"]["user_id"] = uid
    config["configurable"]["user_role"] = role

    # 只取最后一条消息（历史由 checkpointer 按 thread_id 还原，前端每次只发本轮 1 条）
    last = messages_data[-1] if messages_data else {}
    content = last.get("content", "") if isinstance(last, dict) else ""
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    # 长期记忆入库与agent并行
    from back.agent.memory import extract_memories
    # ① 扔到就绪队列，返回 Task对象。
    _mt = asyncio.create_task(extract_memories(uid, content))
    ## ② 防止 GC 误杀（asyncio 内部只持弱引用）
    _memory_tasks.add(_mt)
    ## ③ Task 跑完后自动清理，防 set 无限涨大
    _mt.add_done_callback(_memory_tasks.discard)

    # ── 意图编排：路由 → 槽位检查 → prompt 组装 → Agent 获取 ──
    result = await process_task(
        user_message=content,
        thread_id=thread_id,
        intent_override=intent_override,
    )
    if isinstance(result, QuickReply):
        return _sse_once(result.text)

    agent = result.agent
    msgs = result.messages
    intent = result.intent
    raw_user_msg = result.raw_user_msg

    # ── 语义缓存（仅 RAG 单轮问答）：相似问题命中直接返回 ──
    if intent == "rag":
        from back.service.rag.semantic_cache import query as semcache_query
        cached = await semcache_query(raw_user_msg)
        if cached is not None:
            return _sse_once(cached)

    return StreamingResponse(
        _stream_agent_output(agent, {"messages": msgs}, config=config, intent=intent, raw_user_msg=raw_user_msg, request=request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",#不缓存流内容
            "Connection": "keep-alive",#SSE 长连接不能断
            "X-Accel-Buffering": "no", #避免用户前几秒白屏
        },
    )


# ── HITL 恢复 ──

@router.post("/runs/resume", tags=["任务"])
async def resume_agent(
    body: dict,
    request: Request,
    ctx: UserContext = Depends(get_current_user),
):
    """HITL 中断恢复 — 用户确认/拒绝后继续 agent 执行。"""
    from langgraph.types import Command
    from back.agent.agents import get_agent

    thread_id = body.get("thread_id", "")
    intent = body.get("intent", "")
    action = body.get("action", "approve")

    if not thread_id or not intent:
        raise HTTPException(status_code=400, detail="thread_id 和 intent 不能为空")

    agent = get_agent(intent)
    config = {"configurable": {"thread_id": thread_id}}
    config["configurable"]["user_id"] = ctx.user_id
    config["configurable"]["user_role"] = ctx.role

    resume_data = {"decisions": [{"type": action}]}
    resume = Command(resume=resume_data)

    #_stream_agent_output传入resume代替第一次新用户消息
    return StreamingResponse(
        _stream_agent_output(agent, resume, config=config, intent=intent, raw_user_msg="", request=request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── 会话端点（从 LangGraph checkpointer 读取）──

@router.get("/threads", tags=["会话"])
async def list_threads():
    """列出当前用户的所有会话线程。"""
    from back.agent.agents import list_threads as agent_list_threads
    return await agent_list_threads()


@router.get("/threads/{thread_id}/state", tags=["会话"])
async def get_thread_state(thread_id: str):
    """获取指定会话的完整消息历史（从 LangGraph checkpointer 读取）。"""
    from back.agent.agents import get_thread_state as read_state, serialize_messages_for_frontend

    values = await read_state(thread_id)
    if not values:
        raise HTTPException(status_code=404, detail=f"会话 {thread_id} 不存在")

    messages = values.get("messages", [])
    return {"values": {"messages": serialize_messages_for_frontend(messages)}}


# ── 兼容旧版 LangGraph Studio 端点 ──

@router.get("/assistants/{id}")
async def ga(id: str): return {"assistant_id": id, "graph_id": "agent"}

@router.get("/assistants/{id}/schemas")
async def gs(id: str): return {}

@router.post("/threads/search")
async def search_threads(): return []

@router.post("/assistants/search")
async def al(): return [{"assistant_id": "agent", "graph_id": "agent"}]
