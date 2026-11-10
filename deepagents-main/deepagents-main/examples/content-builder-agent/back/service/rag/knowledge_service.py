"""知识库构建 — PDF 解析 → 分块 → 双写向量库 + BM25 存储。"""

import hashlib
import os

from langchain_core.documents import Document

from back.common.exceptions import ServiceException, NotFoundException
from back.service.rag.pdf import process_pdf
from back.service.rag.splitter import splitter
from back.service.rag.embedder import db
from back.service.rag.bm25_store import save as bm25_save, remove as bm25_remove


def _file_md5(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _doc_exists(file_path: str, user_id: str) -> bool:
    raw = db.get(where={"user_id": user_id, "doc_md5": _file_md5(file_path)})
    return len(raw.get("documents", [])) > 0


def remove_doc(doc_md5: str, user_id: str) -> None:
    raw = db.get(where={"user_id": user_id, "doc_md5": doc_md5})
    if not raw.get("ids"):
        raise NotFoundException("文档不存在")

    file_path = raw["metadatas"][0].get("file_path", "")

    try:
        db.delete(where={"user_id": user_id, "doc_md5": doc_md5})
        bm25_remove(user_id, doc_md5)
    except Exception:
        raise ServiceException(f"文档删除失败: {doc_md5}")

    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass


def knowledge_build(file_path: str, user_id: str) -> dict:
    """索引一个 PDF：去重 → 解析 → 分块 → 双写（向量库 + BM25），失败回滚。"""

    if _doc_exists(file_path, user_id):
        raise ServiceException(f"文件重复上传: {file_path}", http_status=409)

    name = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].replace(".pdf", "")
    md5 = _file_md5(file_path)
    text = process_pdf(file_path)["text"]
    doc = Document(page_content=text, metadata={
        "doc_name": name, "user_id": user_id, "doc_md5": md5, "file_path": file_path,
    })
    chunks = splitter.split_documents([doc])
    items = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]

    try:
        db.add_documents(chunks)
        bm25_save(user_id, items)
    except Exception:
        db.delete(where={"user_id": user_id, "doc_md5": md5})
        raise ServiceException(f"文件 {name} 构建失败，已回滚")

    return {"name": name, "md5": md5, "chunks": len(chunks)}
