"""语义缓存 — 粗筛（Q-Q）+ 精排（Q-A）双阶段判断

原理：
  阶段1 粗筛：
    新问题 vs 缓存中所有旧问题 → embedding 余弦相似度 → Top 3 > 阈值粗
    "这两个问题像不像？" → 快速缩小候选范围

  阶段2 精排：
    新问题 vs 候选旧答案 → embedding 余弦相似度 → max > 阈值精
    "这个旧答案能不能回答新问题？" → 只有答案确实相关才复用

  两阶段都过 → 返回缓存答案
  任一不过 → 返回None → 调 LLM → 写入缓存调用store()

容量控制：LRU，最多 MAX_ENTRIES 条。知识库变更时手动 clear()。
重启后缓存从零开始自然预热。
"""

import logging
import os

import numpy as np

from back.service.rag.embedder import embedder

_log = logging.getLogger("semcache")

COARSE_THRESHOLD = float(os.environ.get("SEMCACHE_COARSE", "0.80"))
FINE_THRESHOLD = float(os.environ.get("SEMCACHE_FINE", "0.85"))
MAX_ENTRIES = int(os.environ.get("SEMCACHE_MAX", "500"))
TOP_K = int(os.environ.get("SEMCACHE_TOPK", "3"))

# 内存索引
_q_embeddings: np.ndarray | None = None   # shape: (N, dim)  旧问题向量
_a_embeddings: np.ndarray | None = None   # shape: (N, dim)  旧答案向量
_answers: list[str] = []                   # 答案文本


def _encode(text: str) -> np.ndarray:
    """对文本做 embedding，返回 L2 归一化向量。"""
    emb = embedder.embed_query(text)
    vec = np.array(emb, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


async def query(question: str) -> str | None:
    """双阶段语义缓存查询。"""
    global _q_embeddings, _a_embeddings, _answers
    if _q_embeddings is None or len(_answers) == 0:
        return None

    q_vec = _encode(question)

    # 阶段1：粗筛 — Q-Q 相似度
    qq_scores = np.dot(_q_embeddings, q_vec)
    top_indices = np.argsort(qq_scores)[::-1][:TOP_K]
    candidates = [(i, float(qq_scores[i])) for i in top_indices if float(qq_scores[i]) >= COARSE_THRESHOLD]

    if not candidates:
        return None

    # 阶段2：精排 — Q-A 相似度
    candidate_indices = [i for i, _ in candidates]
    qa_scores = np.dot(_a_embeddings[candidate_indices], q_vec)

    best_local = int(np.argmax(qa_scores))
    best_idx = candidate_indices[best_local]
    fine_score = float(qa_scores[best_local])
    coarse_score = candidates[best_local][1]

    if fine_score < FINE_THRESHOLD:
        _log.info("SemCache MISS  coarse=%.3f fine=%.3f  q=%.30s...",
                  coarse_score, fine_score, question)
        return None

    answer = _answers[best_idx]

    # LRU：命中项移到队尾
    _move_to_end(best_idx)

    _log.info("SemCache HIT  coarse=%.3f fine=%.3f  q=%.30s...",
              coarse_score, fine_score, question)
    return answer


async def store(question: str, answer: str):
    """写入内存缓存（纯内存，向量只编码一次）。"""
    if not answer or len(answer.strip()) < 10:
        return

    global _q_embeddings, _a_embeddings, _answers
    q_vec = _encode(question)
    a_vec = _encode(answer[:500])

    if _q_embeddings is None:
        _q_embeddings = q_vec.reshape(1, -1)
        _a_embeddings = a_vec.reshape(1, -1)
        _answers = [answer]
    else:
        if len(_answers) >= MAX_ENTRIES:
            _q_embeddings = _q_embeddings[1:]
            _a_embeddings = _a_embeddings[1:]
            _answers = _answers[1:]
        _q_embeddings = np.vstack([_q_embeddings, q_vec.reshape(1, -1)])
        _a_embeddings = np.vstack([_a_embeddings, a_vec.reshape(1, -1)])
        _answers.append(answer)

    _log.debug("SemCache SET  q=%.30s...  total=%d", question, len(_answers))


def clear():
    """清空内存缓存。"""
    global _q_embeddings, _a_embeddings, _answers
    _q_embeddings = None
    _a_embeddings = None
    _answers.clear()
    _log.info("SemCache cleared")


def _move_to_end(idx: int):
    """LRU：命中项移到数组末尾（最近使用，最后被淘汰）。"""
    global _q_embeddings, _a_embeddings, _answers
    if _q_embeddings is None or idx >= len(_answers) - 1:
        return

    q_row = _q_embeddings[idx:idx + 1]
    a_row = _a_embeddings[idx:idx + 1]
    ans = _answers[idx]

    _q_embeddings = np.delete(_q_embeddings, idx, axis=0)
    _a_embeddings = np.delete(_a_embeddings, idx, axis=0)
    _answers.pop(idx)

    _q_embeddings = np.vstack([_q_embeddings, q_row])
    _a_embeddings = np.vstack([_a_embeddings, a_row])
    _answers.append(ans)
