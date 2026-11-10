"""Offline persistence, retrieval and graph integration checks."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage

from src.embeddings import EmbeddingClient
from src.graph import CodingAgent, _messages_to_json
from src.memory import MemoryStore
from src.tools import check_permission
from fake_milvus import FakeMilvus


class FakeEmbeddings:
    identity = "test-provider|multilingual-v1"

    async def embed_text(self, text):
        return [1.0, 0.0] if "uv" in text or "依赖" in text else [0.0, 1.0]


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.embedder = FakeEmbeddings()
        self.client = FakeMilvus()
        self.store = MemoryStore(self.root, self.embedder, self.client)

    async def test_schema_persistence_and_project_isolation(self):
        saved = [await self.store.save_memory({"type": "project", "content": "使用 uv"})]
        reopened = MemoryStore(self.root, self.embedder, self.client)
        self.assertEqual(reopened.list_memories(), saved)
        self.assertEqual(MemoryStore(self.root / "nested", self.embedder, self.client).list_memories(), [])
        info = self.client.describe_collection(self.store.alias)
        self.assertEqual([f["name"] for f in info["fields"]], ["id", "type", "content", "vector"])
        self.assertFalse((self.root / ".claude-code-agent" / "memory.db").exists())

    async def test_index_search_ranking_threshold_and_duplicate(self):
        entries = [{"type": "project", "content": "使用 uv"},
                   {"type": "user", "content": "偏好中文回复"}]
        saved = [await self.store.save_memory(entry) for entry in entries]
        duplicate = [await self.store.save_memory(entries[0])]
        self.assertEqual(duplicate[0].id, saved[0].id)
        results = await self.store.search_memories("如何安装依赖")
        self.assertEqual([r["id"] for r in results], [saved[0].id])
        self.assertAlmostEqual(results[0]["score"], 1)
        self.assertEqual(self.client.last_search["search_params"]["metric_type"], "COSINE")
        self.assertEqual(self.client.last_search["consistency_level"], "Strong")
        self.assertEqual(len(await self.store.search_memories("依赖", threshold=-1)), 2)
        self.assertEqual(len(await self.store.search_memories("依赖", top_k=1, threshold=-1)), 1)

    async def test_update_refreshes_vector_and_delete_removes_entity(self):
        memory = (await self.store.save_memory({"type": "project", "content": "使用 uv"}))
        await self.store.save_memory({"id": memory.id, "type": "user", "content": "偏好中文回复"})
        self.assertEqual(await self.store.search_memories("依赖"), [])
        self.assertEqual((await self.store.search_memories("中文"))[0]["id"], memory.id)
        self.store.delete_memory(memory.id)
        self.assertEqual(self.store.list_memories(), [])
        self.assertEqual(await self.store.search_memories("中文"), [])

    async def test_embedding_failure_never_changes_existing_entity(self):
        memory = (await self.store.save_memory({"type": "project", "content": "使用 uv"}))
        with patch.object(self.embedder, "embed_text", side_effect=RuntimeError("offline")):
            with self.assertRaises(RuntimeError):
                await self.store.save_memory({"id": memory.id, "type": "project", "content": "new"})
        self.assertEqual(self.store.get_memory(memory.id), memory)
        self.assertEqual((await self.store.search_memories("依赖"))[0]["content"], memory.content)

    async def test_failed_first_write_does_not_publish_collection(self):
        self.client.fail_upsert = True
        with self.assertRaises(RuntimeError):
            await self.store.save_memory({"type": "project", "content": "使用 uv"})
        self.assertNotIn(self.store.alias, self.client.aliases)
        self.assertEqual(self.store.list_memories(), [])

    async def test_model_change_requires_reindex_and_swaps_complete_collection(self):
        saved = [await self.store.save_memory({"type": "project", "content": "使用 uv"})]
        old_name = self.client.aliases[self.store.alias]
        self.embedder.identity = "different-model-same-dimensions"
        with self.assertRaisesRegex(ValueError, "reindex"):
            await self.store.search_memories("依赖")
        with self.assertRaisesRegex(ValueError, "reindex"):
            await self.store.save_memory({"type": "user", "content": "中文"})
        self.assertEqual(await self.store.reindex(), 1)
        self.assertNotEqual(self.client.aliases[self.store.alias], old_name)
        self.assertNotIn(old_name, self.client.collections)
        self.assertEqual(self.store.list_memories(), saved)
        self.assertEqual(len(await self.store.search_memories("依赖")), 1)

    async def test_empty_collection_can_change_embedding_model(self):
        saved = [await self.store.save_memory({"type": "project", "content": "使用 uv"})]
        self.store.delete_memory(saved[0].id)
        self.embedder.identity = "new-model"
        self.assertEqual(await self.store.reindex(), 0)
        await self.store.save_memory({"type": "project", "content": "使用 uv"})
        self.assertEqual(len(await self.store.search_memories("依赖")), 1)

    async def test_failed_reindex_keeps_old_alias_and_data(self):
        saved = [await self.store.save_memory({"type": "project", "content": "使用 uv"})]
        old_name = self.client.aliases[self.store.alias]
        self.client.fail_upsert = True
        with self.assertRaises(RuntimeError):
            await self.store.reindex()
        self.assertEqual(self.client.aliases[self.store.alias], old_name)
        self.assertEqual(self.store.list_memories(), saved)

    async def test_reindex_rejects_concurrently_changed_content(self):
        memory = (await self.store.save_memory({"type": "project", "content": "使用 uv"}))

        async def delete_during_embedding(texts):
            self.store.delete_memory(memory.id)
            return [1, 0]

        with patch.object(self.embedder, "embed_text", side_effect=delete_during_embedding):
            with self.assertRaisesRegex(ValueError, "changed during reindex"):
                await self.store.reindex()
        self.assertEqual(self.store.list_memories(), [])

    async def test_update_rejects_concurrent_change_and_search_escapes_text(self):
        memory = (await self.store.save_memory({"type": "project", "content": 'uv "quoted" 中文'}))
        await self.store.save_memory({"type": "project", "content": 'uv "quoted" 中文'})
        self.assertEqual(len(self.store.list_memories()), 1)

        async def delete_during_embedding(texts):
            self.store.delete_memory(memory.id)
            return [1, 0]

        with patch.object(self.embedder, "embed_text", side_effect=delete_during_embedding):
            with self.assertRaisesRegex(ValueError, "Unknown memory ID"):
                await self.store.save_memory({"id": memory.id, "type": "project", "content": "uv new"})

    async def test_invalid_memory_and_vectors_are_rejected(self):
        for entry in ([], {"memories": [{"type": "user", "content": "x"}]},
                      {"type": "invalid", "content": "x"},
                      {"type": "user", "content": " "},
                      {"id": "missing", "type": "user", "content": "x"}):
            with self.assertRaises(ValueError):
                await self.store.save_memory(entry)
        for vector in ([], [0, 0], [float("inf"), 1]):
            with patch.object(self.embedder, "embed_text", return_value=vector):
                with self.assertRaises(ValueError):
                    await self.store.save_memory({"type": "user", "content": "x"})
        self.assertEqual(self.store.list_memories(), [])

    async def test_failed_fact_does_not_undo_other_facts(self):
        with patch.object(self.embedder, "embed_text", wraps=self.embedder.embed_text) as embed:
            first = await self.store.save_memory({"type": "user", "content": "偏好中文回复"})
            embed.side_effect = RuntimeError("offline")
            with self.assertRaises(RuntimeError):
                await self.store.save_memory({"type": "project", "content": "使用 uv"})
            self.assertEqual(self.store.list_memories(), [first])
            embed.side_effect = None
            second = await self.store.save_memory({"type": "project", "content": "使用 uv"})
            self.assertEqual([c.args[0] for c in embed.await_args_list], ["偏好中文回复", "使用 uv", "使用 uv"])
        self.assertEqual({m.id for m in self.store.list_memories()}, {first.id, second.id})

    async def test_empty_project_needs_no_embedding_credentials(self):
        with patch.object(self.embedder, "embed_text", side_effect=AssertionError("should not call")):
            self.assertEqual(await self.store.search_memories("hello"), [])


class EchoModel:
    def __init__(self):
        self.inputs = []

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.inputs.append(messages)
        yield AIMessageChunk(content="done")


class GraphMemoryTests(unittest.IsolatedAsyncioTestCase):
    setUp = MemoryTests.setUp

    async def test_tool_sequence_skips_one_fact_and_saves_the_other(self):
        existing = await self.store.save_memory({"type": "user", "content": "偏好中文回复"})
        agent = object.__new__(CodingAgent)
        agent._memory_store = self.store
        agent._memory_dirty = False
        with patch.object(self.embedder, "embed_text", wraps=self.embedder.embed_text) as embed, \
             patch.object(self.client, "upsert", wraps=self.client.upsert) as upsert:
            result = json.loads(await agent._execute_memory_tool("memory_search", {"query": "偏好中文回复"}))
            self.assertEqual(result[0]["id"], existing.id)
            # An unchanged fact ends after search; only the new fact is saved.
            self.assertEqual(json.loads(await agent._execute_memory_tool("memory_search", {"query": "使用 uv"})), [])
            result = json.loads(await agent._execute_memory_tool("memory_save", {"type": "project", "content": "使用 uv"}))
            self.assertIsInstance(result["saved"], dict)
            self.assertEqual(result["saved"]["content"], "使用 uv")
            self.assertTrue(agent._memory_dirty)
            self.assertEqual([c.args[0] for c in embed.await_args_list], ["偏好中文回复", "使用 uv", "使用 uv"])
            upsert.assert_called_once()
            self.assertEqual(len(upsert.call_args.kwargs["data"]), 1)
            rejected = await agent._execute_memory_tool("memory_save", {"memories": [{"type": "user", "content": "x"}]})
            self.assertTrue(rejected.startswith("Error:"))
            self.assertEqual(upsert.call_count, 1)
        self.assertEqual(self.store.get_memory(existing.id), existing)
        self.assertEqual(len(self.store.list_memories()), 2)

    async def test_first_turn_recall_is_not_checkpointed_and_refreshes_after_writes(self):
        await self.store.save_memory({"type": "project", "content": "使用 uv 管理依赖"})
        model = EchoModel()
        with patch("src.graph.AsyncSqliteSaver", None), \
             patch.object(CodingAgent, "_create_llm", return_value=model), \
             patch("pathlib.Path.cwd", return_value=self.root):
            agent = CodingAgent(custom_system_prompt="base", api_base="https://invalid.test/v1")
        agent._memory_store = self.store
        agent._emit_streaming_chunk = lambda chunk: None
        agent._streaming_emitted = True
        with patch("src.graph.start_spinner"), patch("src.graph.stop_spinner"), \
             patch("src.graph.print_assistant_text"), patch("src.graph.print_cost"):
            result = await agent._graph.ainvoke(
                {"messages": [HumanMessage(content="如何安装依赖")], "permission_mode": "default"},
                {"configurable": {"thread_id": "memory-integration"}},
            )
        self.assertEqual(model.inputs[0][0].content, "base")
        request_user = next(m for m in model.inputs[0] if isinstance(m, HumanMessage))
        self.assertIn("【相关历史记忆】", request_user.content)
        self.assertIn("使用 uv", request_user.content)
        self.assertTrue(request_user.content.endswith("【当前用户问题】\n如何安装依赖"))
        serialized = json.dumps(_messages_to_json(result["messages"]), ensure_ascii=False)
        self.assertNotIn("使用 uv", serialized)
        checkpoint = await agent._graph.aget_state({"configurable": {"thread_id": "memory-integration"}})
        self.assertNotIn("使用 uv", json.dumps(_messages_to_json(checkpoint.values["messages"]), ensure_ascii=False))
        self.assertEqual(next(m.content for m in result["messages"] if isinstance(m, HumanMessage)), "如何安装依赖")
        with patch("src.graph.start_spinner"), patch("src.graph.stop_spinner"), \
             patch("src.graph.print_assistant_text"), patch("src.graph.print_cost"):
            next_result = await agent._graph.ainvoke(
                {"messages": [HumanMessage(content="继续安装依赖")], "permission_mode": "default"},
                {"configurable": {"thread_id": "memory-integration"}},
            )
        request_users = [m for m in model.inputs[-1] if isinstance(m, HumanMessage)]
        self.assertEqual(request_users[0].content, "如何安装依赖")
        self.assertTrue(request_users[-1].content.endswith("【当前用户问题】\n继续安装依赖"))
        self.assertEqual(request_users[-1].content.count("【相关历史记忆】"), 1)
        self.assertNotIn("【相关历史记忆】", json.dumps(_messages_to_json(next_result["messages"]), ensure_ascii=False))
        self.assertEqual(agent._memory_query, "继续安装依赖")
        memory_id = self.store.list_memories()[0].id
        response = await agent._dispatch_execute("memory_save", {
            "id": memory_id, "type": "project", "content": "使用 uv add 安装依赖"
        })
        self.assertIn("saved", response)
        self.assertTrue(agent._memory_dirty)
        await agent._refresh_memory_context()
        self.assertIn("uv add", agent._memory_context)
        await agent._dispatch_execute("memory_delete", {"memory_id": memory_id})
        await agent._refresh_memory_context()
        self.assertEqual(agent._memory_context, "")
        await agent.close()

    async def test_request_copy_targets_current_user_through_tool_loop(self):
        agent = CodingAgent.__new__(CodingAgent)
        agent._memory_message_id = "current-user"
        agent._memory_context = "one fact"
        system = SystemMessage(content="fixed rules")
        old_user = HumanMessage(content="same question", id="old-user")
        current_user = HumanMessage(content="same question", id="current-user", additional_kwargs={"tag": "keep"})
        call = AIMessage(content="", tool_calls=[
            {"name": "memory_search", "args": {"query": "fact"}, "id": "call-1"}
        ])
        tool = ToolMessage(content="tool result", tool_call_id="call-1")
        history = [system, old_user, AIMessage(content="old reply"), current_user, call, tool]
        before = _messages_to_json(history)
        first = agent._messages_with_memory(history)
        second = agent._messages_with_memory(history)
        self.assertEqual(_messages_to_json(history), before)
        self.assertIs(first[0], system)
        self.assertIs(first[1], old_user)
        self.assertIs(first[-1], tool)
        self.assertIsNot(first[3], current_user)
        self.assertEqual(first[3].id, current_user.id)
        self.assertEqual(first[3].additional_kwargs, current_user.additional_kwargs)
        self.assertEqual(first[3].content, second[3].content)
        self.assertEqual(first[3].content.count("【相关历史记忆】"), 1)
        agent._memory_context = "updated fact"
        self.assertNotIn("one fact", agent._messages_with_memory(history)[3].content)
        agent._memory_context = ""
        self.assertEqual(_messages_to_json(agent._messages_with_memory(history)), before)

    async def test_content_blocks_are_preserved_and_missing_current_user_is_skipped(self):
        agent = CodingAgent.__new__(CodingAgent)
        agent._memory_context = "fact"
        agent._memory_message_id = "current"
        blocks = [{"type": "text", "text": "question"},
                  {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}}]
        original = HumanMessage(content=blocks, id="current")
        result = agent._messages_with_memory([original])
        self.assertEqual(result[0].content[1:], blocks)
        self.assertIn("【当前用户问题】", result[0].content[0]["text"])
        self.assertEqual(original.content, blocks)
        old = HumanMessage(content="old", id="old")
        self.assertIs(agent._messages_with_memory([old])[0], old)
        system = SystemMessage(content="fixed")
        self.assertEqual(agent._messages_with_memory([system]), [system])

    async def test_compaction_retains_current_user_before_request_injection(self):
        agent = CodingAgent.__new__(CodingAgent)
        agent._base_system_prompt = "fixed rules"
        agent.permission_mode = "default"
        agent._memory_context = "recalled fact"
        agent._memory_message_id = "current"
        agent._llm = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content="summary")))
        current = HumanMessage(content="current question", id="current")
        history = [SystemMessage(content="fixed rules"), HumanMessage(content="old question"),
                   AIMessage(content="old answer"), HumanMessage(content="another question"),
                   AIMessage(content="another answer"), current]
        with patch("src.graph.print_info"):
            await agent._compact_conversation(history)
        self.assertIs(history[-1], current)
        request = agent._messages_with_memory(history)
        self.assertEqual(request[0].content, "fixed rules")
        self.assertEqual(request[1].content, "[Previous conversation summary]\nsummary")
        self.assertIn("recalled fact", request[-1].content)
        self.assertEqual(history[-1].content, "current question")

    async def test_recall_failure_clears_previous_turn_without_breaking_preprocess(self):
        agent = CodingAgent.__new__(CodingAgent)
        agent._memory_store = self.store
        agent._memory_query = "hello"
        agent._memory_context = "old memory"
        agent._memory_dirty = True
        with patch.object(self.store, "search_memories", side_effect=RuntimeError("offline")), \
             patch("src.graph.print_info"):
            await agent._refresh_memory_context()
        self.assertEqual(agent._memory_context, "")

    async def test_permissions_enforce_readonly_and_confirmation_modes(self):
        with patch("src.tools._check_json_rules", return_value=False):
            self.assertEqual(check_permission("memory_search", {}, "plan")["action"], "allow")
            for name in ("memory_save", "memory_delete"):
                self.assertEqual(check_permission(name, {}, "plan")["action"], "deny")
                self.assertEqual(check_permission(name, {}, "default")["action"], "ask")
                self.assertEqual(check_permission(name, {}, "bypassPermissions")["action"], "allow")
        with patch("src.tools._check_json_rules", return_value=True):
            self.assertEqual(check_permission("memory_search", {}, "bypassPermissions")["action"], "deny")


class EmbeddingConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_config_has_actionable_error(self):
        with patch.dict(os.environ, {}, clear=True):
            client = EmbeddingClient()
        with self.assertRaisesRegex(ValueError, "EMBEDDING_MODEL"):
            await client.embed_text("hello")


if __name__ == "__main__":
    unittest.main()
