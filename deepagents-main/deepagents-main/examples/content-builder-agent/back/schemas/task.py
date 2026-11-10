"""Agent 任务 — 请求/响应模型"""

from typing import Optional
from pydantic import BaseModel, Field


# ── 提交任务：请求 ──

#参数注解：
# BaseModel 子类，这里只解了请求体的input和config
class TaskSubmitRequest(BaseModel):
    """提交 Agent 任务。请求体包含 input（messages 列表）和 config 配置。"""
    input: dict = Field(..., description="Agent 输入，含 messages 列表") #...表示该字段必填，无默认值
    config: Optional[dict] = Field(default_factory=dict, description="configurable 配置（thread_id、intent 等）")

