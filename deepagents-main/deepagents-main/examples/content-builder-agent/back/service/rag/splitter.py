"""智能分块 — 按段落优先，metadata 自动继承。"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from back.common.config import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP

splitter = RecursiveCharacterTextSplitter(
    chunk_size=RAG_CHUNK_SIZE,
    chunk_overlap=RAG_CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", ".", " ", ""],
)
