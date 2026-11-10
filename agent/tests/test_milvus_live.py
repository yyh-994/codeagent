"""Opt-in live Milvus test using synthetic vectors, with scoped cleanup."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from src.memory import MemoryStore


class LiveEmbeddings:
    identity = "synthetic-live-test-v1"

    async def embed_text(self, text):
        return [1.0, 0.0] if "uv" in text else [0.0, 1.0]


@unittest.skipUnless(os.environ.get("RUN_MILVUS_TESTS") == "1", "Set RUN_MILVUS_TESTS=1 for a live server")
class MilvusLiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_crud_isolation_restart_and_reindex(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            embedder = LiveEmbeddings()
            store = MemoryStore(root, embedder)
            other = MemoryStore(root / "nested", embedder)
            try:
                saved = [await store.save_memory(entry) for entry in [
                    {"type": "project", "content": "Use uv for dependencies"},
                    {"type": "user", "content": "Chinese replies"},
                ]]
                uv = saved[0]
                results = await store.search_memories("uv dependencies")
                self.assertEqual([item["id"] for item in results], [uv.id])
                self.assertEqual(await asyncio.to_thread(other.list_memories), [])
                self.assertEqual(len([await store.save_memory({"type": "project", "content": "Use uv for dependencies"})]), 1)
                self.assertEqual(len(await asyncio.to_thread(store.list_memories)), 2)
                await asyncio.to_thread(store.close)
                self.assertEqual(len(await asyncio.to_thread(store.list_memories)), 2)
                await store.save_memory({"id": uv.id, "type": "project", "content": "Use uv add"})
                self.assertEqual((await store.search_memories("uv"))[0]["content"], "Use uv add")
                embedder.identity = "synthetic-live-test-v2"
                with self.assertRaisesRegex(ValueError, "reindex"):
                    await store.search_memories("uv")
                self.assertEqual(await store.reindex(), 2)
                self.assertEqual((await store.search_memories("uv"))[0]["id"], uv.id)
                await asyncio.to_thread(store.delete_memory, uv.id)
                self.assertEqual(await store.search_memories("uv"), [])
            finally:
                for target in (store, other):
                    client = target.client
                    if client.has_collection(collection_name=target.alias):
                        client.drop_alias(alias=target.alias)
                    for name in client.list_collections():
                        if name.startswith(target.alias + "_"):
                            client.drop_collection(collection_name=name)
                    await asyncio.to_thread(target.close)


if __name__ == "__main__":
    unittest.main()
