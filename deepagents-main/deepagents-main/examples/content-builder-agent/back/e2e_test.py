"""端到端测试 — 所有核心功能"""
import asyncio, httpx, sys

asserted = False  # 不因单个断言退出，跑完所有

async def check(name, coro, expect_status=200):
    try:
        r = await coro
        if r.status_code != expect_status:
            print(f"  FAIL {name}: status={r.status_code} body={r.text[:100]}")
            return None
        print(f"  OK   {name}")
        return r
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        return None

async def main():
    print("=== End-to-End Test ===\n")
    c = httpx.AsyncClient(base_url="http://127.0.0.1:2024", timeout=60)

    # 1. Health
    r = await check("Health", c.get("/health"))
    if r: print(f"       body={r.json()}")

    # 2. Login
    r = await check("Login", c.post("/auth/login", json={"username":"admin","password":"admin123"}))
    if not r: await c.aclose(); return
    data = r.json()
    token = data["access_token"]
    print(f"       user={data['user_id'][:8]} role={data['role']}")

    # 3. Logout
    await check("Logout", c.post("/auth/logout"))

    # Re-login
    r = await c.post("/auth/login", json={"username":"admin","password":"admin123"})
    token = r.json()["access_token"]
    h = {"Authorization": f"Bearer {token}"}

    # 5. SSE stream
    print("\n  SSE stream (10s timeout):")
    try:
        events = []
        async with c.stream("POST", "/runs/stream", headers=h, json={
            "input": {"messages": [{"role":"user","content":"1+1=?一句话"}]}
        }) as s:
            i = 0
            async for line in s.aiter_lines():
                if line.startswith("event:") or line.startswith("data:"):
                    events.append(line[:80])
                i += 1
                if "1+1" in line or "2" in line or i > 40:
                    break
        think_events = [e for e in events if "think" in e]
        print(f"       Total {len(events)} SSE lines, {len(think_events)} think events")
        if think_events:
            print(f"       Sample: {think_events[-1][:80]}")
    except Exception as e:
        print(f"       FAIL: {e}")

    # 6. KB search
    await check("KB search", c.get("/rag/search?q=test", headers=h))

    await c.aclose()
    print("\n=== Done ===")

asyncio.run(main())
