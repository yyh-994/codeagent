"""登录/登出路由"""

from fastapi import APIRouter, Request

from back.controller.auth import login, do_logout
from back.schemas import LoginRequest, TokenResponse, MessageResponse

router = APIRouter()


@router.post("/auth/login", response_model=TokenResponse, tags=["认证"])
async def auth_login(body: LoginRequest):
    """用户名密码登录，返回 access token。"""
    return await login(body.username, body.password)


@router.post("/auth/logout", response_model=MessageResponse, tags=["认证"])
async def auth_logout(request: Request):
    """登出——从 Redis 删除 access token。"""
    access_token = request.headers.get("Authorization", "").replace("Bearer ", "")
    await do_logout(access_token)
    return {"msg": "ok"}
