"""Redis 客户端 — 全局异步 client，启动时在 lifespan 中初始化"""

from back.common.config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD

_redis_available = False
_pool = None
client = None

try:
    import redis.asyncio as aioredis
    _redis_available = True
except ImportError:
    aioredis = None  # type: ignore


async def init_redis():
    """启动时建连接池 + 验证连通性。lifespan 中调用。"""
    global _pool, client
    if not _redis_available:
        raise RuntimeError("redis 包未安装，请 pip install redis")
    if _pool is None:
        _pool = aioredis.ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            max_connections=50,
        )
        client = aioredis.Redis(connection_pool=_pool)
        await client.ping()


# Lua 脚本：KEYS[1] = request_path, ARGV[1] = ttl
# Redis 执行 INCR，首次设 EXPIRE，返回计数
_INCR_EXPIRE_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return count
"""


async def aredis_incr_with_expire(request_path: str, ttl: int) -> int:
    """原子 INCR + EXPIRE（只首次设 TTL）。返回计数。"""
    if client is None:
        return 0
    return await client.eval(_INCR_EXPIRE_SCRIPT, 1, request_path, ttl)
