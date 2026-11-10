"""知识库搜索 — 向量 + BM25 混合召回 → Cross-encoder 重排序。"""

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

from back.common.config import RAG_RERANKER_MODEL
from back.common.exceptions import ServiceException
from back.service.rag.embedder import db
from back.service.rag.bm25_store import load as bm25_load

reranker = CrossEncoder(RAG_RERANKER_MODEL)


def search(query: str, user_id: str, top_k: int = 5,
           vector_k: int = 15, bm25_k: int = 10) -> list[str]:
    """混合检索：向量 + BM25 RRF 融合 → cross-encoder 重排 → top_k。"""
    try:
        vector_retriever = db.as_retriever(
            search_kwargs={"k": vector_k, "filter": {"user_id": user_id}},
        )

        items = bm25_load(user_id)
        #list[dict]->list[Document]
        docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in items]
        bm25_retriever = BM25Retriever.from_documents(docs, k=bm25_k)

        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )
        candidates = ensemble.invoke(query)
        if not candidates:
            return []

        pairs = [[query, doc.page_content] for doc in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc.page_content for doc, _ in ranked[:top_k]]
    except Exception as e:
        raise ServiceException(f"知识库检索失败: {e}") from e
