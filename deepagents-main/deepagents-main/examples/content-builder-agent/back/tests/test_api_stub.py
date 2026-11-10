"""测试存根端点 — 无外部依赖，直接验证"""
import pytest


@pytest.mark.asyncio
async def test_ok(client):
    res = await client.get("/ok")
    assert res.status_code == 200
    assert res.json() == {"ok": True}


@pytest.mark.asyncio
async def test_info(client):
    res = await client.get("/info")
    assert res.status_code == 200
    assert res.json() == {"version": "1.0"}


@pytest.mark.asyncio
async def test_root(client):
    res = await client.get("/")
    assert res.status_code == 200
    assert res.json() == {"msg": "Content Builder Agent API"}


@pytest.mark.asyncio
async def test_assistants_by_id(client):
    res = await client.get("/assistants/test-agent")
    assert res.status_code == 200
    data = res.json()
    assert data["assistant_id"] == "test-agent"
    assert data["graph_id"] == "agent"


@pytest.mark.asyncio
async def test_assistants_schemas(client):
    res = await client.get("/assistants/test-agent/schemas")
    assert res.status_code == 200
    assert res.json() == {}


@pytest.mark.asyncio
async def test_threads_empty(client):
    res = await client.get("/threads")
    assert res.status_code == 200
    assert res.json() == []


@pytest.mark.asyncio
async def test_threads_search(client):
    res = await client.post("/threads/search")
    assert res.status_code == 200
    assert res.json() == []


@pytest.mark.asyncio
async def test_assistants_search(client):
    res = await client.post("/assistants/search")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["assistant_id"] == "agent"


@pytest.mark.asyncio
async def test_health(client, mock_redis):
    """健康检查 — DB 会 fail（没有真实 MySQL），Redis mock 通过。"""
    res = await client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["redis"] == "ok"
    # DB 预期 fail（没有真实连接）
    assert data["db"] in ("ok", "fail")
