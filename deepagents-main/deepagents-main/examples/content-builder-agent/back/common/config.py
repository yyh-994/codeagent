"""全局配置中心 — 所有可调参数从 .env 读取，一处管理"""

import os

# ── 服务 ──
SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "2024"))

# ── JWT ──
JWT_SECRET = os.environ.get("JWT_SECRET", "")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_EXPIRE_MINUTES = int(os.environ.get("ACCESS_EXPIRE_MINUTES", "30"))
MAX_LOGIN_ATTEMPTS = int(os.environ.get("MAX_LOGIN_ATTEMPTS", "5"))
LOGIN_LOCKOUT_MINUTES = int(os.environ.get("LOGIN_LOCKOUT_MINUTES", "15"))

# ── RAG ──
RAG_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "500"))
RAG_CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "80"))
RAG_VECTOR_TOP_K = int(os.environ.get("RAG_VECTOR_TOP_K", "5"))
RAG_VECTOR_PRE_FETCH = int(os.environ.get("RAG_VECTOR_PRE_FETCH", "15"))
RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RAG_RERANKER_MODEL = os.environ.get("RAG_RERANKER_MODEL", "BAAI/bge-reranker-large")
VISION_MODEL_NAME = os.environ.get("VISION_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")

# ── 长期记忆（全部集中在此，勿散落别处）──
MEMORY_NS = "memories"  # store namespace 第二段，写入(agents.py)和检索(middleware.py)共用，改必须两侧同时生效
# 检索侧（UserMemoryMiddleware 注入）
MEMORY_TOP_K = int(os.environ.get("MEMORY_TOP_K", "5"))                     # 每轮最多注入几条记忆
MEMORY_MIN_SCORE = float(os.environ.get("MEMORY_MIN_SCORE", "0.4"))        # 低于此相关性不注入
# 入库侧（extract_memories 裁决预筛）
MEMORY_RECONCILE_TOP_K = int(os.environ.get("MEMORY_RECONCILE_TOP_K", "5"))  # 每条新事实检索最相似的 K 条旧记忆送裁决
# 向量化（embedding 模型复用 RAG 的 RAG_EMBEDDING_MODEL，见上方 RAG 段）
MEMORY_EMBEDDING_DIMS = int(os.environ.get("MEMORY_EMBEDDING_DIMS", "512"))  # bge-small-zh-v1.5 输出维度

# ── 模型 ──
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek:deepseek-reasoner")
IMAGE_GEN_MODEL = os.environ.get("IMAGE_GEN_MODEL", "black-forest-labs/FLUX.1-schnell")
SEARCH_MAX_RESULTS = int(os.environ.get("SEARCH_MAX_RESULTS", "5"))
HUGGINGFACE_TOKEN=os.environ.get("HUGGINGFACE_TOKEN")

# ── 日志 ──
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_MAX_MB = int(os.environ.get("LOG_MAX_MB", "10"))
LOG_BACKUPS = int(os.environ.get("LOG_BACKUPS", "5"))
SLOW_REQUEST_MS = int(os.environ.get("SLOW_REQUEST_MS", "3000"))
LOG_FORMAT = os.environ.get("LOG_FORMAT", "text")  # text | json

# ── 数据库 ──
DB_POOL_SIZE = int(os.environ.get("DB_POOL_SIZE", "10"))
DB_POOL_RECYCLE = int(os.environ.get("DB_POOL_RECYCLE", "3600"))
# DATABASE_URL 已移除，db_client.py 改为用 MYSQL_* 分散变量拼 URL

# ── Redis ──
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD") or None

# ── 限流 ──
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "50"))
