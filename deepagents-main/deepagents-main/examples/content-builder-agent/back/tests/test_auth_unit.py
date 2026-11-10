"""测试 auth.py — get_current_user Dependency + 暴力破解防护"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from back.controller.auth import UserContext


class TestGetCurrentUser:
    """get_current_user() — FastAPI Dependency，JWT 解码 + Redis 撤销检查。"""

    @staticmethod
    def _credentials(token: str = "good.token"):
        creds = MagicMock()
        creds.credentials = token
        return creds

    @pytest.mark.asyncio
    async def test_valid_token_returns_user_context(self):
        """合法 token → 返回 UserContext(user_id, role)。"""
        from back.controller.auth import get_current_user
        mock_r = AsyncMock()
        mock_r.exists.return_value = True
        with patch("jose.jwt.decode", return_value={"userid": "uuid-1", "role": "vip"}), \
             patch("back.controller.auth.client", mock_r):
            ctx = await get_current_user(credentials=self._credentials())
        assert isinstance(ctx, UserContext)
        assert ctx.user_id == "uuid-1"
        assert ctx.role == "vip"

    @pytest.mark.asyncio
    async def test_missing_header_raises(self):
        """无 Bearer header → AuthException。"""
        from back.common.exceptions import AuthException
        from back.controller.auth import get_current_user
        with pytest.raises(AuthException, match="缺少认证令牌"):
            await get_current_user(credentials=None)

    @pytest.mark.asyncio
    async def test_decode_error_raises(self):
        """JWT 过期/签名错 → AuthException。"""
        from jose import JWTError
        from back.common.exceptions import AuthException
        from back.controller.auth import get_current_user
        with patch("jose.jwt.decode", side_effect=JWTError("expired")), \
             pytest.raises(AuthException, match="无效的认证令牌"):
            await get_current_user(credentials=self._credentials())

    @pytest.mark.asyncio
    async def test_revoked_token_raises(self):
        """token 不在 Redis 活跃会话中（已登出）→ AuthException。"""
        from back.common.exceptions import AuthException
        from back.controller.auth import get_current_user
        mock_r = AsyncMock()
        mock_r.exists.return_value = False
        with patch("jose.jwt.decode", return_value={"userid": "u", "role": "basic"}), \
             patch("back.controller.auth.client", mock_r), \
             pytest.raises(AuthException, match="令牌已失效"):
            await get_current_user(credentials=self._credentials())

    @pytest.mark.asyncio
    async def test_default_role_when_missing(self):
        """JWT 无 role 字段 → 默认 basic。"""
        from back.controller.auth import get_current_user
        with patch("jose.jwt.decode", return_value={"userid": "uuid-1"}), \
             patch("back.controller.auth.client", AsyncMock()):
            ctx = await get_current_user(credentials=self._credentials())
        assert ctx.role == "basic"


class TestCheckBruteForce:
    @pytest.mark.asyncio
    async def test_not_locked_initially(self, mock_async_db):
        from back.controller.auth import _check_brute_force

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_async_db.execute.return_value = mock_result

        result = await _check_brute_force("testuser")
        assert result is False

    @pytest.mark.asyncio
    async def test_locked_after_max_attempts(self, mock_async_db):
        from back.controller.auth import _check_brute_force, MAX_LOGIN_ATTEMPTS

        fake_attempts = [MagicMock() for _ in range(MAX_LOGIN_ATTEMPTS)]
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = fake_attempts
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_async_db.execute.return_value = mock_result

        result = await _check_brute_force("hacker")
        assert result is True
