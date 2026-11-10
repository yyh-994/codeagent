"""RAG 服务包。

⚠️ 故意不在这里 re-export 子模块：包级 import 会连带加载
search_service 的模块级 CrossEncoder 重排模型（数 GB、可能触发联网下载），
让所有 `from back.service.rag.embedder import ...` 之类的轻量导入被拖死。
全项目调用方均使用完整模块路径导入（如 back.service.rag.search_service）。
"""
