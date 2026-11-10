"""BM25 文档存储 — 按 user_id 分文件，与 ChromaDB 双写。"""

import json
import os

STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output", "bm25")
os.makedirs(STORE_DIR, exist_ok=True)


def load(user_id: str) -> list[dict]:
    p = os.path.join(STORE_DIR, f"{user_id}.json")
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

"""
json文件内容
[
  {"page_content": "文档1切片1的内容...", "metadata": {"doc_name": "文档1", "user_id": "xxx", "doc_md5": "abc", "file_path": "..."}},
  {"page_content": "文档1切片2的内容...", "metadata": {"doc_name": "文档1", "user_id": "xxx", "doc_md5": "abc", "file_path": "..."}},
  {"page_content": "文档2切片1的内容...", "metadata": {"doc_name": "文档2", "user_id": "xxx", "doc_md5": "def", "file_path": "..."}},
  {"page_content": "文档2切片2的内容...", "metadata": {"doc_name": "文档2", "user_id": "xxx", "doc_md5": "def", "file_path": "..."}}
]
"""
def save(user_id: str, chunks: list[dict]) -> None:
    data = load(user_id)
    data.extend(chunks)
    with open(os.path.join(STORE_DIR, f"{user_id}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def remove(user_id: str, doc_md5: str) -> None:
    data = load(user_id)
    data = [d for d in data if d["metadata"].get("doc_md5") != doc_md5]
    with open(os.path.join(STORE_DIR, f"{user_id}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
