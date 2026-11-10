"""中间件 — 请求日志 + 慢请求告警 + 请求限流

认证通过 FastAPI Dependency（back.controller.auth.get_current_user）完成，
用户身份通过 config.configurable 传递。
"""
import time, uuid, logging
from contextvars import ContextVar
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from back.common.config import SLOW_REQUEST_MS, RATE_LIMIT_PER_MIN

logger = logging.getLogger("request")
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

RATE_LIMIT_WHITELIST = {"/health", "/health/live", "/health/ready", "/ok"}


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = str(uuid.uuid4())[:8]
        request_id_var.set(rid)

        start = time.time()
        response = await call_next(request)
        elapsed = int((time.time() - start) * 1000)

        logger.info(
            "%s %s → %s | %sms | %s",
            request.method, request.url.path,
            response.status_code, elapsed,
            request.client.host if request.client else "-",
        )
        if elapsed > SLOW_REQUEST_MS:
            logger.warning(
                "SLOW REQUEST | %s %s | %sms",
                request.method, request.url.path, elapsed,
            )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """固定窗口限流 — 按 IP + 端点维度，Redis INCR + EXPIRE，Retry-After 响应头"""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in RATE_LIMIT_WHITELIST:
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        key = f"rate:{ip}:{path}"

        try:
            from back.repository.redis_client import aredis_incr_with_expire
            count = await aredis_incr_with_expire(key, 60)
            if count == 0:  # Redis 不可用
                return await call_next(request)

            limit = RATE_LIMIT_PER_MIN
            if count > limit:
                return JSONResponse(
                    status_code=429,
                    content={"code": "RATE_LIMITED", "msg": "请求过于频繁，请稍后再试", "data": {}},
                    headers={"Retry-After": "60", "X-RateLimit-Limit": str(limit)},
                )
        except Exception:
            pass  # Redis 不可用时放行
        return await call_next(request)

