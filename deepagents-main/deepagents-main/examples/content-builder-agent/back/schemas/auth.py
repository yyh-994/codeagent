"""Pydantic 请求/响应模型 — 输入校验 + API 文档"""

from pydantic import BaseModel, Field


# ── 认证 ──

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50, description="用户名")
    password: str = Field(..., min_length=1, max_length=128, description="密码")


class TokenResponse(BaseModel):
    access_token: str = Field(..., description="JWT access token")


class MessageResponse(BaseModel):
    msg: str = "ok"
