"""Offline sentence embeddings for long-term memory."""

from __future__ import annotations

import asyncio
import math
import os
import threading
from pathlib import Path


class EmbeddingClient:
    def __init__(self):
        self.model = os.environ.get("EMBEDDING_MODEL", "").strip()
        self.local_path = os.environ.get("EMBEDDING_LOCAL_PATH", "").strip()
        self.device = os.environ.get("EMBEDDING_DEVICE", "cpu").strip()
        self._local_model = None
        self._local_lock = threading.Lock()

    @property
    def identity(self) -> str:
        path = Path(self.local_path or self.model).expanduser().resolve()
        return f"local|sentence-transformers|{path}"

    async def embed_text(self, text: str) -> list[float]:
        if not self.model:
            raise ValueError("Configure EMBEDDING_MODEL for the local embedding model.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding input must be nonempty text.")
        vector = await asyncio.to_thread(self._embed_local, text)
        return self._validate_vector(vector)

    def _embed_local(self, text: str):
        path = Path(self.local_path or self.model).expanduser().resolve()
        if not path.is_dir():
            raise ValueError("Local embedding model not found. Set EMBEDDING_LOCAL_PATH "
                             "to an existing model snapshot directory; no download will be attempted.")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ValueError('Local embeddings require local dependencies: pip install -e ".[local]"') from None
        # Loading/encoding can outlive a cancelled async caller; serialize on a thread lock.
        with self._local_lock:
            if self._local_model is None:
                self._local_model = SentenceTransformer(
                    str(path), device=self.device, local_files_only=True,
                    trust_remote_code=False, token=False,
                )
            return self._local_model.encode(
                text, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )

    @staticmethod
    def _validate_vector(result):
        values = result.tolist() if hasattr(result, "tolist") else result
        if not isinstance(values, list) or not values:
            raise ValueError("Expected one sentence-level embedding vector.")
        try:
            vector = [float(value) for value in values]
        except (TypeError, ValueError):
            raise ValueError("Expected one sentence-level embedding vector.") from None
        norm = math.hypot(*vector)
        if not all(math.isfinite(value) for value in vector) or not math.isfinite(norm) or norm == 0:
            raise ValueError("Embedding vector must be finite and nonzero.")
        return vector
