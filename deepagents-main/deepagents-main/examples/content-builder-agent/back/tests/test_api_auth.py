"""测试认证端点 — POST /auth/login, /auth/refresh, /auth/logout"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestAuthLogin:
    """POST /auth/login — 注：成功/失败路径需要完整 DB mock，
    当前因 alembic 等依赖缺失，仅测试参数校验层。"""

    @pytest.mark.asyncio
    async def test_login_success(self, client, mock_async_db, mock_redis):
        pytest.skip("需要完整 DB mock（alembic 未安装）")

    @pytest.mark.asyncio
    async def test_login_bad_credentials(self, client, mock_async_db, mock_redis):
        pytest.skip("需要完整 DB mock（alembic 未安装）")

    @pytest.mark.asyncio
    async def test_login_empty_username(self, client):
        """空用户名 → 422（Pydantic 参数校验层拦截）。"""
        res = await client.post("/auth/login", json={"username": "", "password": "123"})
        assert res.status_code == 422

    @pytest.mark.asyncio
    async def test_login_empty_password(self, client):
        """空密码 → 422。"""
        res = await client.post("/auth/login", json={"username": "admin", "password": ""})
        assert res.status_code == 422


class TestAuthRefresh:
    """POST /auth/refresh — 注：需要完整 DB mock，当前仅测试参数校验。"""

    @pytest.mark.asyncio
    async def test_refresh_success(self, client, mock_async_db, mock_redis):
        pytest.skip("需要完整 DB mock（alembic 未安装）")

    @pytest.mark.asyncio
    async def test_refresh_invalid_token(self, client, mock_async_db, mock_redis):
        pytest.skip("需要完整 DB mock（alembic 未安装）")


class TestAuthLogout:
    """POST /auth/logout — 注：需要完整 DB mock，当前仅测试参数校验。"""

    @pytest.mark.asyncio
    async def test_logout_ok(self, client, mock_async_db, mock_redis):
        pytest.skip("需要完整 DB mock（alembic 未安装）")

    @pytest.mark.asyncio
    async def test_logout_no_token(self, client):
        res = await client.post("/auth/logout")
        assert res.status_code == 200
