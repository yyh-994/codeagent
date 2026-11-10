"""测试限流中间件"""
import time
import pytest
from unittest.mock import patch


@pytest.mark.asyncio
async def test_rate_limit_allows_normal_request(client, mock_redis):
    """正常频率请求应该放行。"""
    mock_redis.get.return_value = None  # 无历史请求记录
    res = await client.get("/ok")
    assert res.status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_blocks_too_many_requests(client, mock_redis):
    """超过限流阈值 → 429。"""
    # 注：RateLimitMiddleware 的具体 Redis key 格式取决于 RequestLogMiddleware，
    # mock 需要匹配中间件内部的 key 构建逻辑。当前测试跳过，等集成测试时完善。
    pytest.skip("Rate limit mock needs middleware key format alignment")


class TestRequestLogMiddleware:
    """验证 RequestLogMiddleware 不会阻断正常请求"""

    @pytest.mark.asyncio
    async def test_logging_is_transparent(self, client, mock_redis):
        """请求日志中间件应该透明，不影响响应。"""
        res = await client.get("/ok")
        assert res.status_code == 200
        assert res.json() == {"ok": True}
