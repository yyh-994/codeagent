"""测试 POST /runs/stream SSE 端点 — mock Agent 输出"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessageChunk, ToolMessage


def _make_ai_chunk(content: str = "", tool_chunks: list | None = None) -> AIMessageChunk:
    """构造一个模拟的 AIMessageChunk。"""
    return AIMessageChunk(
        content=content,
        tool_call_chunks=tool_chunks or [],
        id="test-msg-id",
    )


def _make_tool_msg(name: str, content: str = "") -> ToolMessage:
    """构造一个模拟的 ToolMessage。"""
    return ToolMessage(content=content, name=name, tool_call_id="tc-1")


class TestStreamAgent:
    """POST /runs/stream — 通过 mock agent.astream 测试 SSE 输出"""

    @pytest.mark.asyncio
    async def test_text_only_stream(self, auth_client):
        """纯文本回复 → 产生 think 事件。"""
        fake_chunks = [
            (_make_ai_chunk("你好"), {}),
            (_make_ai_chunk("，世界"), {}),
        ]

        fake_agent = MagicMock()
        fake_agent.astream.return_value = _async_gen(fake_chunks)

        with patch("back.agent.intent.route_intent", new=AsyncMock(return_value={"intent": "write", "topic": "测试主题"})), \
             patch("back.agent.agents.get_agent", return_value=fake_agent):
            res = await auth_client.post(
                "/runs/stream",
                json={
                    "input": {"messages": [{"role": "user", "content": "hi"}]},
                    "config": {"configurable": {"thread_id": "th-test"}},
                },
            )
            assert res.status_code == 200
            assert "text/event-stream" in res.headers.get("content-type", "")
            body = res.text
            assert "event: think" in body
            assert "你好" in body
            assert "世界" in body

    @pytest.mark.asyncio
    async def test_tool_call_stream(self, auth_client):
        """工具调用 → 产生 tool + think 事件。"""
        fake_chunks = [
            (
                _make_ai_chunk(
                    content="我来搜索一下",
                    tool_chunks=[{"name": "web_search", "id": "tc-1", "index": 0}],
                ),
                {},
            ),
            (_make_ai_chunk(content="。"), {}),
            (_make_tool_msg("web_search", "搜索结果..."), {}),
            (_make_ai_chunk("搜索完成，答案是..."), {}),
        ]

        fake_agent = MagicMock()
        fake_agent.astream.return_value = _async_gen(fake_chunks)

        with patch("back.agent.intent.route_intent", new=AsyncMock(return_value={"intent": "write", "topic": "测试主题"})), \
             patch("back.agent.agents.get_agent", return_value=fake_agent):
            res = await auth_client.post(
                "/runs/stream",
                json={"input": {"messages": [{"role": "user", "content": "搜索"}]}},
            )
            assert res.status_code == 200
            body = res.text
            assert "event: tool" in body
            assert "web_search" in body
            assert "event: think" in body

    @pytest.mark.asyncio
    async def test_image_gen_event(self, auth_client):
        """生图工具 → 产生 tool_result 事件。"""
        fake_chunks = [
            (_make_ai_chunk("生成封面", [{"name": "generate_image", "id": "tc-img", "index": 0}]), {}),
            (_make_tool_msg("generate_image", "Image saved"), {}),
        ]

        fake_agent = MagicMock()
        fake_agent.astream.return_value = _async_gen(fake_chunks)

        with patch("back.agent.intent.route_intent", new=AsyncMock(return_value={"intent": "write", "topic": "测试主题"})), \
             patch("back.agent.agents.get_agent", return_value=fake_agent):
            res = await auth_client.post(
                "/runs/stream",
                json={"input": {"messages": [{"role": "user", "content": "生成封面"}]}},
            )
            assert res.status_code == 200
            assert "event: tool_result" in res.text
            assert "generate_image" in res.text

    @pytest.mark.asyncio
    async def test_file_content_event(self, auth_client):
        """文件写入工具 → 产生 tool_result 事件。"""
        fake_chunks = [
            (_make_ai_chunk("写博客", [{"name": "write_file", "id": "tc-wf", "index": 0}]), {}),
            (_make_tool_msg("write_file", "# 博客正文内容"), {}),
        ]

        fake_agent = MagicMock()
        fake_agent.astream.return_value = _async_gen(fake_chunks)

        with patch("back.agent.intent.route_intent", new=AsyncMock(return_value={"intent": "write", "topic": "测试主题"})), \
             patch("back.agent.agents.get_agent", return_value=fake_agent):
            res = await auth_client.post(
                "/runs/stream",
                json={"input": {"messages": [{"role": "user", "content": "写一篇博客"}]}},
            )
            assert res.status_code == 200
            assert "event: tool_result" in res.text
            assert "博客正文内容" in res.text

    @pytest.mark.asyncio
    async def test_empty_messages_fails(self, auth_client):
        """空消息 → 400。"""
        res = await auth_client.post(
            "/runs/stream",
            json={"input": {"messages": []}},
        )
        assert res.status_code == 400

    @pytest.mark.asyncio
    async def test_stream_error_yields_error_event(self, auth_client):
        """Agent 异常 → 产生 error 事件。"""
        fake_agent = MagicMock()

        async def _error_gen():
            yield (_make_ai_chunk("开始"), {})
            raise RuntimeError("模型崩溃")

        fake_agent.astream.return_value = _error_gen()

        with patch("back.agent.intent.route_intent", new=AsyncMock(return_value={"intent": "write", "topic": "测试主题"})), \
             patch("back.agent.agents.get_agent", return_value=fake_agent):
            res = await auth_client.post(
                "/runs/stream",
                json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
            )
            assert res.status_code == 200
            assert "event: error" in res.text

    @pytest.mark.asyncio
    async def test_requires_auth(self, client):
        """不带 token → 401。"""
        res = await client.post(
            "/runs/stream",
            json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
        )
        assert res.status_code == 401


def _async_gen(items: list):
    """将 list 转为 async generator。"""
    async def gen():
        for item in items:
            yield item
    return gen()
