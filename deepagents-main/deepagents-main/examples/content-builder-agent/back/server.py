#!/usr/bin/env python3
"""Content Builder Agent — 生产级 FastAPI 入口"""

import os, sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)

# ⚠️ 先加载 .env，再 import 其他模块（config.py 需要读环境变量）
from dotenv import load_dotenv
load_dotenv(os.path.join(_project_root, ".env"))

from back.common.logging import setup_logging
from back.common.config import SERVER_HOST, SERVER_PORT, LOG_LEVEL
setup_logging()

import logging, signal
_log = logging.getLogger("server")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from back.common.middleware import RequestLogMiddleware, RateLimitMiddleware
from back.common.exception_handler import register_exception_handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期。关键组件失败直接退出，非关键组件降级。"""
    import asyncio as _asyncio

    # ── 必须：DB ──
    from back.repository.db_client import init_db
    await _asyncio.wait_for(init_db(), timeout=10)
    _log.info("DB ready")

    # ── 必须：Agent 异步存储（checkpointer + store）──
    from back.agent.agents import init_async_stores
    await _asyncio.wait_for(init_async_stores(), timeout=30)
    _log.info("Agent stores ready")

    # ── Redis ──
    from back.repository.redis_client import init_redis
    await _asyncio.wait_for(init_redis(), timeout=5)
    _log.info("Redis ready")

    # ── 预热 Agent 缓存（失败不阻塞，首次请求慢一些）──
    try:
        from back.agent.agents import get_agent
        from back.agent.school_mcp import get_query_tools
        await get_query_tools()
        for intent in ("write", "rag", "query"):
            get_agent(intent)
        _log.info("Agent cache warmed")
    except Exception as e:
        _log.warning("Agent warmup skipped: %s", e)

    # ── 预热 RAG 管线（失败不阻塞）──
    try:
        from back.service.rag.splitter import splitter       # noqa: F811, F401
        from back.service.rag.embedder import embedder, db   # noqa: F811, F401
        from back.service.rag.search_service import _reranker  # noqa: F811, F401
        _log.info("RAG models loaded")
    except Exception as e:
        _log.warning("RAG warmup skipped: %s", e)



    yield
    # ── shutdown ──
    _log.info("Shutting down...")
    from back.repository.redis_client import _pool as redis_pool
    if redis_pool:
        await redis_pool.disconnect()
    from back.repository.db_client import engine as db_engine
    await db_engine.dispose()

app = FastAPI(title="Content Builder Agent", version="1.0", lifespan=lifespan)

# ── 中间件（越后加的越外层，越先拦截请求）：
# 每次请求 → CORSMiddleware → RateLimitMiddleware → RequestLogMiddleware → 路由
app.add_middleware(RequestLogMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
register_exception_handler(app)

# ── 健康检查（K8s 标准）──

@app.get("/health")
async def health():
    """汇总健康状态。"""
    from back.repository.redis_client import client
    try:
        if client: await client.ping(); redis = "ok"
        else: redis = "not_initialized"
    except: redis = "fail"  # noqa: E722
    return {"status": "ok", "redis": redis}

@app.get("/health/live")
async def liveness():
    """K8s liveness probe — 进程是否存活（不查外部依赖）。"""
    return {"alive": True}

@app.get("/health/ready")
async def readiness():
    """K8s readiness probe — 是否可以接流量（检查 Redis）。"""
    from back.repository.redis_client import client
    try:
        if client: await client.ping()
        return {"ready": True}
    except Exception:
        return {"ready": False}

# ── 路由 ──
#让浏览器能直接访问 agent 生成的文件。
app.mount("/output", StaticFiles(directory=os.path.join(_project_root, "back", "output")), name="output")

from back.controller.routes import router as api_router
app.include_router(api_router)

from back.controller.auth_routes import router as auth_router
app.include_router(auth_router)

from back.controller.rag_controller import router as rag_router
app.include_router(rag_router)


if __name__ == "__main__":
    import uvicorn
    # 直接传 app 对象（而不是 "back.server:app" 字符串），避免模块重复加载
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level=LOG_LEVEL.lower(),
    )
