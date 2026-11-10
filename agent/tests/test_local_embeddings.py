"""Local-only loading configuration and opt-in offline model verification."""

import asyncio
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.embeddings import EmbeddingClient


class LocalEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_model_is_cached_and_identity_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {
                "EMBEDDING_MODEL": "BAAI/bge-small-zh-v1.5", "EMBEDDING_LOCAL_PATH": directory,
            }, clear=True):
                client = EmbeddingClient()
            self.assertEqual(client.identity, f"local|sentence-transformers|{Path(directory).resolve()}")
            model = Mock()
            model.encode.return_value = np.array([1.0, 0.0])
            factory = Mock(return_value=model)
            module = SimpleNamespace(SentenceTransformer=factory)
            with patch.dict("sys.modules", {"sentence_transformers": module}):
                self.assertEqual(await client.embed_text("query"), [1, 0])
                self.assertEqual(await client.embed_text("document"), [1, 0])
            factory.assert_called_once_with(str(Path(directory).resolve()), device="cpu",
                                            local_files_only=True, trust_remote_code=False, token=False)
            model.encode.assert_called_with("document", normalize_embeddings=True,
                                             convert_to_numpy=True, show_progress_bar=False)

    async def test_batch_inputs_and_invalid_vectors_are_rejected(self):
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "local-test"}, clear=True):
            client = EmbeddingClient()
        with patch.object(client, "_embed_local") as encode:
            for text in ([], ["one", "two"], "", " ", None):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    await client.embed_text(text)
            encode.assert_not_called()
            for vector in ([], [0, 0], [float("nan"), 1], [[1, 2]]):
                encode.return_value = vector
                with self.subTest(vector=vector), self.assertRaises(ValueError):
                    await client.embed_text("fact")

    async def test_missing_local_path_fails_without_online_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {
                "EMBEDDING_MODEL": "BAAI/bge-small-zh-v1.5",
                "EMBEDDING_LOCAL_PATH": str(Path(directory) / "missing"),
            }, clear=True):
                client = EmbeddingClient()
            with patch("socket.socket.connect", side_effect=AssertionError("Online fallback")):
                with self.assertRaisesRegex(ValueError, "no download"):
                    await client.embed_text("query")


@unittest.skipUnless(os.environ.get("RUN_LOCAL_EMBEDDING_TESTS") == "1", "Set RUN_LOCAL_EMBEDDING_TESTS=1")
class LocalModelLiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_bge_generates_sentence_vectors_without_network(self):
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
        client = EmbeddingClient()
        with patch("socket.socket.connect", side_effect=AssertionError("Local embedding attempted network access")):
            vectors = [await client.embed_text(text) for text in ["本项目使用 uv 管理依赖。", "用户喜欢中文回复。"]]
            query = await client.embed_text("项目依赖用什么工具管理？")
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(query), 512)
        self.assertTrue(all(len(vector) == 512 for vector in vectors))
        self.assertAlmostEqual(math.hypot(*query), 1, places=5)
        scores = [sum(a * b for a, b in zip(query, vector)) for vector in vectors]
        self.assertGreater(scores[0], scores[1])
        if os.environ.get("RUN_MILVUS_TESTS") == "1":
            from src.memory import MemoryStore

            with tempfile.TemporaryDirectory() as directory:
                store = MemoryStore(Path(directory), client)
                try:
                    saved = [await store.save_memory(entry) for entry in [
                        {"type": "project", "content": "本项目使用 uv 管理 Python 依赖。"},
                        {"type": "user", "content": "用户喜欢中文回复。"},
                    ]]
                    results = await store.search_memories("项目依赖用什么工具管理？", threshold=0)
                    self.assertEqual(results[0]["id"], saved[0].id)
                    self.assertIn("uv", results[0]["content"])
                    await asyncio.to_thread(store.delete_memory, saved[0].id)
                    self.assertEqual(len(await asyncio.to_thread(store.list_memories)), 1)
                finally:
                    milvus = store.client
                    if milvus.has_collection(collection_name=store.alias):
                        milvus.drop_alias(alias=store.alias)
                    for name in milvus.list_collections():
                        if name.startswith(store.alias + "_"):
                            milvus.drop_collection(collection_name=name)
                    store.close()
