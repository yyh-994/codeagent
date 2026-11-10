# ── Content Builder Agent — 后端多阶段构建 ──
FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖缓存层
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" || true
RUN pip install --no-cache-dir deepagents fastapi uvicorn[standard] \
    huggingface-hub langchain-deepseek pillow pymysql pyyaml sqlalchemy \
    google-search-results langgraph-checkpoint-sqlite python-jose[cryptography] \
    pymupdf chromadb langchain-text-splitters sentence-transformers \
    langchain-community python-pptx python-multipart pydantic-settings sentry-sdk

# 源码
COPY back/ ./back/
COPY skills/ ./skills/
COPY langgraph.json .
COPY .env .

# 创建运行时目录
RUN mkdir -p /app/output /app/logs /app/data /app/back/data/memory /app/uploads

ENV PYTHONUNBUFFERED=1
EXPOSE 2024

CMD ["uvicorn", "back.server:app", "--host", "0.0.0.0", "--port", "2024"]
