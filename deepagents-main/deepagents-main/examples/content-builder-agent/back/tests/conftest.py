"""pytest 全局 fixture — 环境变量 + mock + test app"""
import os, sys, tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── 确保项目根在 sys.path ──
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ── 环境变量 (在任何 back 模块 import 之前设置) ──
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-testing")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("MYSQL_DATABASE", "test_db")
os.environ.setdefault("MYSQL_USER", "test")
os.environ.setdefault("MYSQL_PASSWORD", "test")
os.environ.setdefault("MYSQL_HOST", "localhost")
os.environ.setdefault("REDIS_HOST", "localhost")


# ── Mock agents.py 模块（避免 SQLite 连接 + 加速测试）──
# routes.py lazy-imports from agents.py, 测试中所有 agent 操作都应被 mock
# 注入一个假模块到 sys.modules，防止真实 agents.py 的模块级副作用执行

def _make_mock_checkpointer():
    """构建一个模拟的 AsyncSqliteSaver，返回空结果。"""
    cp = MagicMock()
    cp.setup = AsyncMock()
    # conn.execute → 返回空 cursor（模拟空数据库）
    mock_cursor = AsyncMock()
    mock_cursor.fetchall = AsyncMock(return_value=[])
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    cp.conn = MagicMock()
    cp.conn.execute = MagicMock(return_value=mock_ctx)
    return cp

_mock_agents = MagicMock()
_mock_agents.get_agent = MagicMock()
_mock_agents.get_checkpointer = _make_mock_checkpointer
_mock_agents.get_thread_state = AsyncMock(return_value={})
_mock_agents.serialize_messages_for_frontend = MagicMock(return_value=[])
_mock_agents.list_threads = AsyncMock(return_value=[])
sys.modules["back.agent.agents"] = _mock_agents

# 意图路由已拆到 back.agent.intent，单独 mock（防止测试真调 LLM）
_mock_intent = MagicMock()
_mock_intent.route_intent = AsyncMock(return_value={"intent": "write", "topic": "测试主题"})
sys.modules["back.agent.intent"] = _mock_intent

# 长期记忆写入侧已拆到 back.agent.memory，单独 mock
_mock_memory = MagicMock()
_mock_memory.extract_memories = AsyncMock()
sys.modules["back.agent.memory"] = _mock_memory

# 不 mock back.agent package — tools 等子模块需要正常导入


# ── 阻止 init_db 副作用 ──
try:
    from back.repository.db_client import init_db as _real_init_db
except Exception:
    _real_init_db = None

if _real_init_db is not None:
    from back.repository import db_client
    db_client.init_db = lambda: None


# ── 工具函数 ──

def make_test_token(payload: dict | None = None, expired: bool = False) -> str:
    """快速生成测试 JWT。"""
    from jose import jwt
    from back.common.config import JWT_SECRET, JWT_ALGORITHM
    p = {
        "sub": "test-uuid-001",
        "role": "vip",
        "jti": "test-jti-001",
    }
    if payload:
        p.update(payload)
    if not expired:
        p["exp"] = 9999999999
    return jwt.encode(p, JWT_SECRET, algorithm=JWT_ALGORITHM)


# ── Fixtures ──


@pytest.fixture
def valid_token() -> str:
    """一个有效的测试 JWT。"""
    return make_test_token()


@pytest.fixture
def app():
    """构建测试 FastAPI app（不含 DB/Redis 副作用）。"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from back.controller.routes import router as api_router
    from back.controller.auth_routes import router as auth_router
    from back.controller.rag_controller import router as rag_router

    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def _test_lifespan(app):
        yield  # 测试不连 DB/Redis

    test_app = FastAPI(title="Test Content Builder Agent", version="1.0", lifespan=_test_lifespan)
    test_app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    # ── 测试版认证依赖：绕过 JWT + Redis，直接返回固定用户 ──
    from back.controller.auth import UserContext

    async def _test_get_current_user():
        return UserContext(user_id="test-uuid-001", role="vip")

    from back.controller.auth import get_current_user
    test_app.dependency_overrides[get_current_user] = _test_get_current_user

    # ── 健康检查 ──
    @test_app.get("/health")
    async def health():
        return {"status": "ok", "db": "ok", "redis": "ok"}

    test_app.include_router(api_router)
    test_app.include_router(auth_router)
    test_app.include_router(rag_router)

    # ── 异常处理器 ──
    from back.common.exception_handler import register_exception_handler
    register_exception_handler(test_app)

    return test_app


@pytest.fixture
async def client(app):
    """httpx 异步测试客户端。"""
    from httpx import ASGITransport, AsyncClient
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def auth_client(client, valid_token):
    """带有效 Authorization header 的测试客户端。"""
    client.headers["Authorization"] = f"Bearer {valid_token}"
    return client


@pytest.fixture
def mock_async_db():
    """Mock AsyncSessionLocal 返回一个假 async session。"""
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.add = MagicMock()

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None

    mock_db = MagicMock()
    mock_db.AsyncSessionLocal = MagicMock(return_value=mock_ctx)
    mock_db.init_db = MagicMock()

    with patch.dict("sys.modules", {"back.repository.db_client": mock_db}):
        yield mock_session


@pytest.fixture
def mock_redis():
    """Mock Redis 异步 client。"""
    mock_r = AsyncMock()
    mock_r.ping = AsyncMock(return_value=True)
    mock_r.exists = AsyncMock(return_value=False)
    mock_r.get = AsyncMock(return_value=None)
    mock_r.setex = AsyncMock()
    mock_r.delete = AsyncMock()
    mock_r.eval = AsyncMock(return_value=1)

    with patch("back.repository.redis_client.client", mock_r):
        yield mock_r
