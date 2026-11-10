# -*- coding: utf-8 -*-
"""临时验证脚本:向量长期记忆(冒烟/正例/空信息不入库/去重)。验证后删除。"""
import asyncio
import os
import sys
import tempfile

_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)

# 加载 .env(与 server.py 相同逻辑)
env_path = os.path.join(_project_root, ".env")
if os.path.exists(env_path):
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.split("#")[0]
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

import back.agent.agents as agents  # noqa: E402

# 用临时目录,不碰真实 agent_store.db
agents._memory_dir = tempfile.mkdtemp(prefix="mem_verify_")
print("temp store dir:", agents._memory_dir)


async def count(store, uid):
    items = await store.asearch((uid, "memories"), limit=100)
    return len(items), [i.value.get("text") for i in items]


async def main():
    await agents.init_async_stores()
    store = agents._store

    # ── 1. 向量库冒烟:直接 put + 语义检索,score 非空 ──
    await store.aput(("t", "memories"), "k1", {"text": "用户叫小明"})
    await store.aput(("t", "memories"), "k2", {"text": "用户喜欢打篮球"})
    hits = await store.asearch(("t", "memories"), query="我的名字是什么", limit=2)
    print("[1] 冒烟:", [(h.value["text"], round(h.score, 3) if h.score else h.score) for h in hits])
    assert hits and hits[0].score is not None, "向量检索无 score"
    assert hits[0].value["text"] == "用户叫小明", "语义排序不符合预期"

    # ── 2. 正例:提取应入库 ──
    await agents.extract_memories("testuser", "我叫小明,是一名研一学生,写代码示例时喜欢用Python")
    n1, texts = await count(store, "testuser")
    print(f"[2] 正例入库 {n1} 条:", texts)
    assert n1 > 0, "正例未入库"

    # ── 3. 无信息可存 → 不入库 ──
    await agents.extract_memories("testuser", "你好,今天天气怎么样?")
    n2, _ = await count(store, "testuser")
    print(f"[3] 空信息消息后新增 {n2 - n1} 条(期望 0)")
    assert n2 == n1, "空信息消息不应入库"

    # ── 4. 去重:同一句话再提取一次,条数不变 ──
    await agents.extract_memories("testuser", "我叫小明,是一名研一学生,写代码示例时喜欢用Python")
    n3, _ = await count(store, "testuser")
    print(f"[4] 重复消息后新增 {n3 - n2} 条(期望 0)")
    assert n3 == n2, "重复记忆未被去重"

    # ── 5. 语义检索正例内容 ──
    hits = await store.asearch(("testuser", "memories"), query="用户是谁,偏好什么语言", limit=5)
    print("[5] 语义检索:", [(h.value["text"], round(h.score, 3)) for h in hits])

    print("\nALL PASSED")


asyncio.run(main())
