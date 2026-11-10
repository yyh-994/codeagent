"""嵌入模型 + ChromaDB 连接。"""

import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from back.common.config import RAG_EMBEDDING_MODEL

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output", "chroma_db")

embedder = HuggingFaceEmbeddings(
    model_name=RAG_EMBEDDING_MODEL,
    cache_folder=os.path.join(CHROMA_DIR, ".cache"),
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedder,
)
