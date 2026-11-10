"""JWT 认证 — access token + 登出 + 防刷 + Dependency"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import Depends
from fastapi.security import HTTPBearer
from jose import jwt
from sqlalchemy import select

from back.common.exceptions import (
    AuthException,
    LoginFailedException, RateLimitException,
)
from back.common.config import (
    JWT_SECRET, JWT_ALGORITHM, ACCESS_EXPIRE_MINUTES,
    MAX_LOGIN_ATTEMPTS, LOGIN_LOCKOUT_MINUTES,
)


@dataclass
class UserContext:
    """认证结果，由 ``get_current_user`` Dependency 注入。"""
    user_id: str
    role: str


async def get_current_user(
    credentials=Depends(HTTPBearer(auto_error=False)),
) -> UserContext:
    """FastAPI Dependency：解码 JWT + Redis 撤销检查 → 返回 UserContext。

    在需要认证的路由上声明 ``ctx: UserContext = Depends(get_current_user)`` 即可。
    未认证/令牌失效时抛出 ``AuthException``。
    """
    if credentials is None:
        raise AuthException("缺少认证令牌")

    token = credentials.credentials

    from jose import JWTError
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise AuthException("无效的认证令牌")

    # 检查 Redis 中 token 是否被撤销（登出后失效）
    from back.repository.redis_client import client
    if client is not None and not await client.exists(f"access:{token}"):
        raise AuthException("令牌已失效")

    return UserContext(
        user_id=payload["userid"],
        role=payload.get("role"),
    )


# ── Redis ──

async def _store_access_token(token: str, ttl_seconds: int) -> None:
    """access token 写入 Redis 标记有效；登出时 DEL 撤销。"""
    try:
        from back.repository.redis_client import client
        if client is not None:
            await client.setex(f"access:{token}", ttl_seconds, "1")
    except Exception:
        pass


# ── 登录审计 ──

async def _record_attempt(username: str, ip: str, success: bool):
    from back.repository.db_client import AsyncSessionLocal
    from back.repository.entity import LoginAttempt
    async with AsyncSessionLocal() as s:
        s.add(LoginAttempt(username=username, ip=ip, success=success))
        try:
            await s.commit()
        except Exception:
            pass


async def _check_brute_force(username: str) -> bool:
    """检查是否被锁定。返回 True = 已锁定"""
    from back.repository.db_client import AsyncSessionLocal
    from back.repository.entity import LoginAttempt
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=LOGIN_LOCKOUT_MINUTES)
    async with AsyncSessionLocal() as s:
        stmt = (
            select(LoginAttempt).where(
                LoginAttempt.username == username,
                LoginAttempt.success.is_(False),                  # SQLAlchemy 版 IS FALSE,含 NULL
                LoginAttempt.created_at > cutoff,
            )
        )
        result = await s.execute(stmt)
        return len(result.scalars().all()) >= MAX_LOGIN_ATTEMPTS


# ── 对外接口 ──

async def login(username: str, password: str, ip: str = "") -> dict:
    """验证用户名密码，成功返回 access token。"""
    if await _check_brute_force(username):
        raise RateLimitException("登录尝试过多，请 15 分钟后重试")

    from sqlalchemy.orm import selectinload
    from back.repository.db_client import AsyncSessionLocal
    from back.repository.entity import User

    async with AsyncSessionLocal() as s:
        stmt = select(User).options(selectinload(User.role_rel)).where(User.username == username)
        result = await s.execute(stmt)
        user = result.scalars().first()

    if not user or not user.check_password(password):
        await _record_attempt(username, ip, False)
        raise LoginFailedException()

    await _record_attempt(username, ip, True)

    role = user.role_rel.role if user.role_rel else "basic"
    now = datetime.now(timezone.utc)
    ttl = ACCESS_EXPIRE_MINUTES * 60

    payload = {"userid": user.uuid, "role": role,
               "exp": now + timedelta(minutes=ACCESS_EXPIRE_MINUTES)}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    await _store_access_token(token, ttl)

    return {"access_token": token}


async def do_logout(access_token: str = ""):
    """登出——从 Redis 删除 access token。"""
    if access_token:
        try:
            from back.repository.redis_client import client
            if client is not None:
                await client.delete(f"access:{access_token}")
        except Exception:
            pass

