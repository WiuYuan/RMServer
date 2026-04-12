
# src/handlers/book_handlers.py
# DOC-BEGIN id=book/handlers#1 type=module v=1
# summary: 书籍阅读模块的核心处理器，负责扫描已有PDF、页面图片提取、LLM注释生成与读取。
#   用户手动将PDF放到books/子目录下，book_scan扫描后自动注册。
#   book_id = PDF相对于books/的路径（去掉.pdf后缀），例如 physics/stats/统计学基础。
#   meta信息存储在 books/{book_id}/ 目录下（与PDF同名文件夹），含meta.json、pages/、annotation.json。
#   页面通过PyMuPDF(fitz)转为PNG图片，存入pages/子目录。
#   注释由多模态LLM生成，输出结构化JSON数组[{position: 0.0~1.0, explanation: "..."}]。
# intent: 无需上传PDF，用户直接管理books目录下的文件；book_scan递归扫描所有.pdf文件并注册。
#   复用BlogGenConfig和_new_llm做LLM调用，复用_extract_json_array做JSON容错解析。
import os
import json
import logging
import traceback
from typing import Optional, List
from pathlib import Path

from src.models.book_requests import (
    BookListReq, BookGetReq, BookDeleteReq,
    BookGetPagesReq, BookGenerateAnnotationReq, BookGetAnnotationReq,
)

# DOC-BEGIN id=book/handlers/imports#1 type=dependency v=1
# summary: 导入PyMuPDF(fitz)用于PDF页面渲染为图片；导入BlogGenConfig和_extract_json_array
#   复用现有博客生成模块的LLM配置和JSON解析能力
# intent: PyMuPDF(fitz)是纯本地PDF处理库，无需外部API（区别于Adobe PDF Services），
#   可直接将PDF指定页面渲染为PNG图片。_extract_json_array提供多级容错JSON解析。
import fitz  # PyMuPDF
from src.utils.article_blog_generator import BlogGenConfig, _new_llm, _extract_json_array

logger = logging.getLogger(__name__)

# PDF文件直接放在books目录下，meta存放在与PDF同名的文件夹中
# 例如: books/physics/stats/统计学基础.pdf  →  book_id = "physics/stats/统计学基础"
#   meta:   books/physics/stats/统计学基础/meta.json
#   pages:  books/physics/stats/统计学基础/pages/page_1.png
BOOKS_ROOT = "books"


def _ensure_books_root():
    os.makedirs(BOOKS_ROOT, exist_ok=True)


# DOC-BEGIN id=book/handlers/paths#1 type=function v=2
# summary: 书籍路径计算——book_id是PDF相对于books/目录的路径（去掉.pdf后缀）。
#   例如 book_id="physics/stats/统计学基础" 对应：
#     PDF文件:     books/physics/stats/统计学基础.pdf
#     meta目录:    books/physics/stats/统计学基础/
#     meta.json:   books/physics/stats/统计学基础/meta.json
#     pages/:      books/physics/stats/统计学基础/pages/
#     annotation:  books/physics/stats/统计学基础/annotation.json
# intent: PDF文件用户手动放置，meta目录由程序自动创建（与PDF同名），
#   这样books目录下每个.pdf文件旁边都有一个同名文件夹存放处理结果。
def _book_pdf_path(book_id: str) -> str:
    """返回PDF文件路径: books/{book_id}.pdf"""
    return os.path.join(BOOKS_ROOT, book_id + ".pdf")


def _book_meta_dir(book_id: str) -> str:
    """返回meta目录路径: books/{book_id}/"""
    return os.path.join(BOOKS_ROOT, book_id)


def _book_meta_path(book_id: str) -> str:
    """返回meta.json路径"""
    return os.path.join(_book_meta_dir(book_id), "meta.json")


def _book_pages_dir(book_id: str) -> str:
    """返回页面图片目录路径"""
    return os.path.join(_book_meta_dir(book_id), "pages")


def _book_annotation_path(book_id: str) -> str:
    """返回注释文件路径"""
    return os.path.join(_book_meta_dir(book_id), "annotation.json")


def _book_page_path(book_id: str, page_num: int) -> str:
    """返回指定页码的PNG图片路径"""
    return os.path.join(_book_pages_dir(book_id), f"page_{page_num}.png")


def _book_id_from_pdf_path(pdf_rel_path: str) -> str:
    """从PDF相对路径计算book_id，去掉.pdf后缀"""
    if pdf_rel_path.endswith(".pdf"):
        return pdf_rel_path[:-4]
    return pdf_rel_path
# DOC-END id=book/handlers/paths#1


# DOC-BEGIN id=book/handlers/load-meta#1 type=function v=2
# summary: meta.json的读写操作。_load_meta返回字典或None，_save_meta确保meta目录存在后写入。
# intent: _save_meta时自动创建meta目录（含pages/子目录），因为meta目录是由程序创建的，
#   不像PDF文件是用户手动放置的。
# DOC-BEGIN id=book/handlers/load-meta#1 type=function v=2
# summary: meta.json的读写操作。_load_meta返回字典或None，_save_meta确保meta目录存在后写入。
# intent: _save_meta时自动创建meta目录（含pages/子目录）。
import time

def _load_meta(book_id: str) -> Optional[dict]:
    meta_path = _book_meta_path(book_id)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_meta(book_id: str, meta: dict):
    meta_dir = _book_meta_dir(book_id)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(_book_pages_dir(book_id), exist_ok=True)
    with open(_book_meta_path(book_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
# DOC-END id=book/handlers/load-meta#1


# ============================================================
# Page image extraction (PyMuPDF)
# ============================================================

# DOC-BEGIN id=book/handlers/extract-page#1 type=function v=1
# summary: 使用PyMuPDF将PDF指定页面渲染为PNG图片，保存到pages/目录。
#   page_num为1-based页码，150 DPI PNG，已存在则跳过。
# intent: 纯本地处理，无需外部API，150DPI约1240x1754像素，足够LLM清晰阅读。
def _extract_page_image(book_id: str, page_num: int) -> str:
    pdf_path = _book_pdf_path(book_id)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pages_dir = _book_pages_dir(book_id)
    os.makedirs(pages_dir, exist_ok=True)
    page_png = _book_page_path(book_id, page_num)
    if os.path.exists(page_png):
        return page_png
    doc = fitz.open(pdf_path)
    try:
        idx = page_num - 1
        if idx < 0 or idx >= len(doc):
            raise ValueError(f"Page {page_num} out of range (total {len(doc)} pages)")
        page = doc[idx]
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = page.get_pixmap(matrix=mat)
        pix.save(page_png)
        logger.info(f"[Book] Extracted page {page_num} -> {page_png}")
        return page_png
    finally:
        doc.close()
# DOC-END id=book/handlers/extract-page#1


# DOC-BEGIN id=book/handlers/extract-pages#1 type=function v=1
# summary: 批量提取页面图片，返回base64编码列表和有效页码列表
# intent: 为LLM注释生成准备输入，只处理有效页码
def _extract_pages_as_b64(book_id: str, pages: list[int]) -> tuple[list[str], list[int]]:
    import base64
    meta = _load_meta(book_id)
    total_pages = meta.get("total_pages", 0) if meta else 0
    valid_pages = [p for p in pages if 1 <= p <= total_pages]
    if not valid_pages:
        return [], []
    images_b64 = []
    for page_num in valid_pages:
        png_path = _extract_page_image(book_id, page_num)
        with open(png_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            images_b64.append(f"data:image/png;base64,{b64}")
    return images_b64, valid_pages
# DOC-END id=book/handlers/extract-pages#1


# ============================================================
# Handlers
# ============================================================

# DOC-BEGIN id=book/handlers/scan#1 type=handler v=1
# summary: 扫描books目录下所有.pdf文件，为没有meta目录的PDF自动注册（读取页数创建meta.json）。
#   book_id = PDF相对于books/的路径去掉.pdf后缀。
# intent: 用户手动管理PDF文件，后端只需扫描并注册，无需上传。
def handle_book_scan() -> dict:
    if not os.path.isdir(BOOKS_ROOT):
        return {"ok": True, "scanned": 0, "books": []}

    registered = 0
    for pdf_file in Path(BOOKS_ROOT).rglob("*.pdf"):
        pdf_rel = str(pdf_file.relative_to(BOOKS_ROOT))
        book_id = _book_id_from_pdf_path(pdf_rel)
        meta_path = _book_meta_path(book_id)

        if os.path.exists(meta_path):
            continue  # 已注册

        doc = fitz.open(str(pdf_file))
        try:
            total_pages = len(doc)
        finally:
            doc.close()

        title = pdf_file.stem
        meta = {
            "book_id": book_id,
            "title": title,
            "total_pages": total_pages,
            "pdf_path": pdf_rel,
            "created_at": time.time(),
        }
        _save_meta(book_id, meta)
        registered += 1
        logger.info(f"[Book] Registered '{book_id}' ({total_pages} pages)")

    return handle_book_list()
# DOC-END id=book/handlers/scan#1


# DOC-BEGIN id=book/handlers/list#1 type=handler v=2
# summary: 列出所有已注册书籍（有meta.json的），返回book_id、title、total_pages
# intent: 扫描BOOKS_ROOT下所有子目录的meta.json汇总
def handle_book_list() -> dict:
    if not os.path.isdir(BOOKS_ROOT):
        return {"ok": True, "books": []}
    books = []
    for dirpath, dirnames, filenames in os.walk(BOOKS_ROOT):
        if "meta.json" in filenames:
            meta_path = os.path.join(dirpath, "meta.json")
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                book_id = meta.get("book_id", "")
                books.append({
                    "book_id": book_id,
                    "title": meta.get("title", ""),
                    "total_pages": meta.get("total_pages", 0),
                    "created_at": meta.get("created_at", 0),
                })
            except Exception:
                continue
    return {"ok": True, "books": books}
# DOC-END id=book/handlers/list#1


# DOC-BEGIN id=book/handlers/get#1 type=handler v=1
# summary: 获取单本书籍完整元数据，含是否有注释
def handle_book_get(book_id: str) -> dict:
    meta = _load_meta(book_id)
    if not meta:
        return {"ok": False, "error": f"Book not found: {book_id}"}
    meta["has_annotation"] = os.path.exists(_book_annotation_path(book_id))
    return {"ok": True, **meta}
# DOC-END id=book/handlers/get#1


# DOC-BEGIN id=book/handlers/delete#1 type=handler v=2
# summary: 删除书籍的meta目录（保留PDF文件）
# intent: 只删除程序生成的meta目录，PDF是用户手动放置的不应删除
def handle_book_delete(book_id: str) -> dict:
    meta_dir = _book_meta_dir(book_id)
    if not os.path.isdir(meta_dir):
        return {"ok": False, "error": f"Book meta not found: {book_id}"}
    import shutil
    shutil.rmtree(meta_dir)
    logger.info(f"[Book] Deleted book meta {book_id}")
    return {"ok": True, "book_id": book_id}
# DOC-END id=book/handlers/delete#1


# DOC-BEGIN id=book/handlers/get-pages#1 type=handler v=1
# summary: 获取指定页面图片，返回base64编码列表
def handle_book_get_pages(book_id: str, pages: list[int]) -> dict:
    meta = _load_meta(book_id)
    if not meta:
        return {"ok": False, "error": f"Book not found: {book_id}"}
    total_pages = meta.get("total_pages", 0)
    import base64
    result_pages = []
    for page_num in pages:
        if page_num < 1 or page_num > total_pages:
            continue
        try:
            png_path = _extract_page_image(book_id, page_num)
            with open(png_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            result_pages.append({"page": page_num, "image": f"data:image/png;base64,{b64}"})
        except Exception as e:
            logger.warning(f"[Book] Failed to extract page {page_num}: {e}")
    return {"ok": True, "pages": result_pages}
# DOC-END id=book/handlers/get-pages#1


# DOC-BEGIN id=book/handlers/generate-annotation#1 type=handler v=1
# summary: 书籍注释生成——将指定页码渲染为图片发给多模态LLM，按position(0~1)逐段注释。
async def handle_book_generate_annotation(
    book_id: str, pages: list[int], model_name: str, api_key: str, llm_url: Optional[str] = None,
) -> dict:
    meta = _load_meta(book_id)
    if not meta:
        return {"ok": False, "error": f"Book not found: {book_id}"}
    total_pages = meta.get("total_pages", 0)
    title = meta.get("title", "Untitled")
    valid_pages = sorted([p for p in pages if 1 <= p <= total_pages])
    if not valid_pages:
        return {"ok": False, "error": "No valid pages specified"}
    logger.info(f"[Book][{book_id}] Generating annotation for pages {valid_pages}...")
    images_b64, valid_pages = _extract_pages_as_b64(book_id, valid_pages)
    if not images_b64:
        return {"ok": False, "error": "Failed to extract page images"}
    page_count = len(images_b64)
    page_range_str = f"{valid_pages[0]}-{valid_pages[-1]}" if len(valid_pages) > 1 else str(valid_pages[0])

    prompt = f"""
你是一位书籍阅读助手。以下是一本书的第{page_range_str}页（共{page_count}张图片），书名为《{title}》。

请仔细阅读这些页面，对每一页中的每一个段落或逻辑单元生成一条注释。

输出要求：严格返回JSON数组，不要输出任何其他内容。格式如下：
[
  {{
    "page": 页码数字,
    "position": 0.0到1.0之间的浮点数,
    "explanation": "这段内容在讲什么"
  }}
]

position 规则：
- 表示该段落在当前页面中的**垂直位置**（归一化到0.0~1.0）
- 0.0 = 页面最顶部，1.0 = 页面最底部
- 每页至少生成3-5条注释，确保覆盖页面所有重要内容

explanation 规则：
- 用中文详细解释"从这个位置到下一个位置之间的内容在讲什么"
- 包含关键概念、公式含义、逻辑关系
- 如果是公式，用 $$...$$ 格式重写并解释每个符号

注意：
- 页码从{valid_pages[0]}开始，到{valid_pages[-1]}结束
- 每页的position独立计算（都是0.0~1.0）
- 严格按照JSON格式输出，不要添加任何额外解释
"""

    config = BlogGenConfig(model_name=model_name, api_key=api_key, llm_url=llm_url)
    llm = _new_llm(config)
    logger.info(f"[Book][{book_id}] Calling multimodal LLM with {page_count} page images...")
    raw_output = llm.query_multimodal(prompt, images_b64, verbose=True)

    try:
        annotations = _extract_json_array(raw_output)
    except Exception as e:
        logger.error(f"[Book][{book_id}] Failed to parse: {e}")
        return {"ok": False, "error": f"LLM output parsing failed: {str(e)}"}

    clean = []
    for item in annotations:
        if not isinstance(item, dict):
            continue
        page, pos, expl = item.get("page"), item.get("position"), item.get("explanation", "")
        if page is None or pos is None or not expl:
            continue
        if int(page) not in valid_pages:
            continue
        try:
            pos = max(0.0, min(1.0, float(pos)))
        except (ValueError, TypeError):
            continue
        clean.append({"page": int(page), "position": round(pos, 3), "explanation": expl.strip()})
    clean.sort(key=lambda x: (x["page"], x["position"]))

    ann_path = _book_annotation_path(book_id)
    existing = []
    if os.path.exists(ann_path):
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    merged = [a for a in existing if a.get("page") not in valid_pages] + clean
    merged.sort(key=lambda x: (x["page"], x["position"]))
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    logger.info(f"[Book][{book_id}] Saved {len(merged)} annotations (added {len(clean)})")
    return {"ok": True, "book_id": book_id, "pages": valid_pages, "annotations_added": len(clean), "annotations_total": len(merged)}
# DOC-END id=book/handlers/generate-annotation#1


# DOC-BEGIN id=book/handlers/get-annotation#1 type=handler v=1
# summary: 读取annotation.json返回全部注释
def handle_book_get_annotation(book_id: str) -> dict:
    meta = _load_meta(book_id)
    if not meta:
        return {"ok": False, "error": f"Book not found: {book_id}"}
    ann_path = _book_annotation_path(book_id)
    if not os.path.exists(ann_path):
        return {"ok": True, "book_id": book_id, "annotations": []}
    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    return {"ok": True, "book_id": book_id, "annotations": annotations}
# DOC-END id=book/handlers/get-annotation#1
