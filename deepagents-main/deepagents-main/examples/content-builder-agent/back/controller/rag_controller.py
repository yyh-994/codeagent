"""RAG 路由 — PDF 上传 + 知识库 CRUD 管理"""
import asyncio, os
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException

from back.controller.auth import get_current_user, UserContext
from back.common.config import MAX_UPLOAD_MB
from back.common.exceptions import ValidationException, FileTooLargeException

router = APIRouter(prefix="/rag", tags=["知识库"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── PDF 上传 ──

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    ctx: UserContext = Depends(get_current_user),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise ValidationException("仅支持 PDF 文件")
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise FileTooLargeException(MAX_UPLOAD_MB)
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(content)
    from back.service.rag.knowledge_service import knowledge_build
    result = await asyncio.to_thread(knowledge_build, path, ctx.user_id)
    return result


# ── 知识库 CRUD ──
@router.delete("/docs/{doc_md5}")
async def delete_doc(
    doc_md5: str,
    ctx: UserContext = Depends(get_current_user),
):
    from back.service.rag.knowledge_service import remove_doc
    remove_doc(doc_md5, ctx.user_id)
