"""PDF 一站式处理 — 文本提取 + 内嵌图片视觉描述。"""

import base64
import os
import re

import pymupdf
from huggingface_hub import InferenceClient

from back.common.config import VISION_MODEL_NAME


def _clean(text: str) -> str:
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def process_pdf(file_path: str) -> dict:
    """提取 PDF 文本 + 图片视觉描述，返回合并后的文本。

    Returns:
        {"text": str}
    """
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_TOKEN not set")

    doc = pymupdf.open(file_path)
    client = InferenceClient(token=token)
    texts: list[str] = []
    images: list[str] = []
    full_texts = ""
    full_images = ""
    try:
        # ── 文本提取 ──
        for page in doc:
            text = _clean(page.get_text())
            if text.strip():
                texts.append(text)
        if texts:
            full_texts = "\n\n".join(texts)

        # ── 图片视觉描述 ──
        for i, page in enumerate(doc):
            for j, img in enumerate(page.get_images(full=True)):
                pix = pymupdf.Pixmap(doc, img[0])
                if pix.n < 5:
                    png_str = base64.b64encode(pix.tobytes("png")).decode()
                    try:
                        desc = client.image_to_text(image=png_str, model=VISION_MODEL_NAME)
                        desc = desc if isinstance(desc, str) else str(desc)
                    except Exception:
                        desc = "图片无法解析"
                    images.append(f"第{i+1}页-图{j+1}: {desc}")
        if images:
            full_images = "\n\n图片描述\n" + "\n".join(images)
    finally:
        doc.close()

    return {"text": f"{full_texts}\n\n{full_images}"}
