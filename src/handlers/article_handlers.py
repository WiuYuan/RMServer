
# src/handlers/article_handlers.py
import os
import json
import asyncio
import traceback
import hashlib
import logging

from src.config import DATA_DIR, ARTICLES_ROOT
from src.utils.file_utils import (
    ensure_dir, read_text_file, write_text_file,
)
from src.utils.article_helpers import (
    resolve_article_abs_path, build_articles_tree,
    clean_html_for_injection, clean_html_for_view, load_raw_html,
    article_txt_path_from_html, article_lock_path, article_error_path,
    article_queued_path,
)
from src.utils.image_extractor import (
    _build_css_var_map, _extract_real_image_data,
    _get_img_size, _get_size_from_data_uri,
)
from src.models.requests import (
    ArticleListReq, ArticleGetHtmlReq, ArticleDeleteBlogReq,
    ArticleExtractImagesReq, TaskSelectArticleReq, ArticleGenerateBlogReq,
)
from src.services.agents import Tool_Calls
from src.utils.article_blog_generator import generate_blog_from_article_tree, BlogGenConfig
from src.services.blog_job_queue import BLOG_JOB_QUEUE
from bs4 import BeautifulSoup
import glob
import zipfile
import re
import tempfile
import io
from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
# DOC-BEGIN id=handlers/articles/adobe-imports#1 type=dependency v=1
# summary: 导入Adobe PDF Services SDK依赖和图片处理库，用于PDF图片提取功能
# intent: PDF图片提取需要Adobe SDK（pdfservices-sdk）和PIL库（Pillow）处理图片；
#   base64用于图片编码，io用于字节流处理，tempfile用于临时文件管理
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import ExtractRenditionsElementType
import base64
from PIL import Image

# 图片处理常量（可调整以平衡质量和token消耗）
PDF_IMAGE_SCALE_FACTOR = 0.5  # 图片缩放比例（50%）

logger = logging.getLogger(__name__)
# DOC-END id=handlers/articles/adobe-imports#1


# DOC-BEGIN id=handlers/articles/cancel-all-blogs#1 type=behavior v=1
# summary: 遍历 ARTICLES_ROOT 下所有 .lock / .queued / .error 文件并删除，
#   清空 BLOG_JOB_QUEUE 中尚未开始的任务；正在执行的 worker 无法强制中断，
#   但其 finally 块会因 .lock 已被删除而正常退出；返回删除的文件列表和错误列表
# intent: 用户点击 "Cancel All" 后，所有中间状态（running/queued/failed）被清除，
#   文章只剩下两种状态：completed（.txt 存在）或 none（什么都没有）；
#   BLOG_JOB_QUEUE.clear() 清空队列防止待执行任务继续启动；
#   当前正在跑的 worker 会在 finally 中发现 .lock 已不存在，不影响逻辑正确性
async def handle_article_cancel_all_blogs():
    root = os.path.abspath(ARTICLES_ROOT)
    removed = []
    errors = []

    # DOC-BEGIN id=handlers/articles/cancel-all-blogs-filter#1 type=behavior v=1
    # summary: 遍历匹配的文件时跳过包含".tts."的文件，避免误删TTS状态文件
    # intent: .tts.lock/.tts.queued/.tts.error文件名也以.lock/.queued/.error结尾，
    #   glob模式会同时匹配到；blog cancel_all不应影响TTS状态，两者是独立的生命周期
    for pattern in ("**/*.lock", "**/*.queued", "**/*.error"):
        for fpath in glob.glob(os.path.join(root, pattern), recursive=True):
            if ".tts." in os.path.basename(fpath):
                continue
            try:
                os.remove(fpath)
                removed.append(fpath)
            except Exception as e:
                errors.append({"path": fpath, "error": str(e)})

    BLOG_JOB_QUEUE.clear()

    return {
        "ok": True,
        "status": "cancelled",
        "removed_count": len(removed),
        "removed": removed,
        "errors": errors,
    }
# DOC-END id=handlers/articles/cancel-all-blogs#1


async def handle_article_get_html(data: ArticleGetHtmlReq):
    abs_path = resolve_article_abs_path(data.article_id)
    title, html = load_raw_html(abs_path)
    return {"ok": True, "article_id": data.article_id, "title": title, "clean_html": clean_html_for_view(html, title)}

async def inject_article_to_task(task_id: str, article_id: str, title: str, content: str):
    """把文章内容注入到 LLM 上下文"""
    task_dir = f"{DATA_DIR}/tasks/{task_id}"
    ensure_dir(task_dir)
    tc = Tool_Calls(LOG_DIR=task_dir, MAX_CHAR=800000, mode="Summary")

    new_tool_calls = [
        {
            "role": "assistant",
            "content": (
                f"[System Context Injection]\n"
                f"User has selected an external article.\n"
                f"Title: {title}\n"
                f"Filename: {article_id}\n\n"
                f"Content Start:\n"
                f"{content[:60000]}\n"
                f"Content End.\n"
            )
        }
    ]
    tc.extend(new_tool_calls)

# === 业务 Handlers ===
async def handle_article_delete_blog(data: ArticleDeleteBlogReq):
    abs_path = resolve_article_abs_path(data.article_id)

    txt_path   = article_txt_path_from_html(abs_path)
    lock_path  = article_lock_path(abs_path)
    err_path   = article_error_path(abs_path)
    queue_path = article_queued_path(abs_path)
    all_img_path = abs_path + ".images.json"

    removed = []
    errors = []

    def _rm(p: str):
        try:
            if os.path.exists(p):
                os.remove(p)
                removed.append(p)
        except Exception as e:
            errors.append({"path": p, "error": str(e)})

    _rm(txt_path)
    _rm(all_img_path)
    _rm(err_path)
    _rm(lock_path)
    _rm(queue_path)

    return {
        "ok": True,
        "status": "deleted",
        "article_id": data.article_id,
        "removed": removed,
        "errors": errors,
    }

# DOC-BEGIN id=handlers/articles/extract-images#1 type=api v=2
# summary: 处理文章图片提取请求，根据文件扩展名（.pdf或.html）分派到不同的处理函数；
#   PDF文件调用Adobe PDF Extract API，HTML文件使用原有的BeautifulSoup解析逻辑
# intent: 统一入口函数，支持PDF和HTML两种格式的图片提取，保持返回格式一致
async def handle_article_extract_images(data: ArticleExtractImagesReq):
    abs_path = resolve_article_abs_path(data.article_id)
    
    # 根据文件扩展名判断处理方式
    if abs_path.lower().endswith('.pdf'):
        # 调用PDF处理函数
        return await handle_article_extract_images_from_pdf(data)
    else:
        # 原有的HTML处理逻辑
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_html = f.read()

        css_var_map = _build_css_var_map(raw_html)

        soup = BeautifulSoup(raw_html, "lxml")

        images_data = []
        seen_src_hashes = set()
        count = 1

        MIN_WIDTH = 100
        MIN_HEIGHT = 100
        MIN_TOTAL = 500

        for img in soup.find_all("img"):
            real_src = _extract_real_image_data(img, css_var_map)
            if not real_src:
                continue

            clean_src = real_src.strip()

            src_hash = hashlib.md5(clean_src.encode()).hexdigest()
            if src_hash in seen_src_hashes:
                continue

            w, h = _get_size_from_data_uri(clean_src)

            print(f"[IMAGE] width={w}, height={h}, total={w+h}")
            if w is not None and h is not None:
                if w < MIN_WIDTH or h < MIN_HEIGHT or w + h < MIN_TOTAL:
                    continue

            if w is None or h is None:
                w2, h2 = _get_img_size(img)
                w = w if w is not None else w2
                h = h if h is not None else h2

            header_part = clean_src.split(",", 1)[0].lower()
            if "image/jpeg" in header_part or "image/jpg" in header_part:
                ext, mime = "jpg", "image/jpeg"
            elif "image/webp" in header_part:
                ext, mime = "webp", "image/webp"
            elif "image/png" in header_part:
                ext, mime = "png", "image/png"
            elif "image/gif" in header_part:
                ext, mime = "gif", "image/gif"
            else:
                continue

            alt_text = (img.get("alt") or "").strip()
            caption = alt_text if alt_text else f"Figure {count}"

            seen_src_hashes.add(src_hash)
            images_data.append({
                "index": count,
                "filename": f"figure_{count}.{ext}",
                "caption": caption,
                "width": w,
                "height": h,
                "mime_type": mime,
                "base64_content": clean_src,
            })
            count += 1

        print(f"[extract_images] css_var_map={len(css_var_map)} images={len(images_data)}")

        return {"ok": True, "article_id": data.article_id, "total": len(images_data), "images": images_data}
# DOC-END id=handlers/articles/extract-images#1

# DOC-BEGIN id=handlers/articles/extract-images-from-pdf#1 type=api v=1
# summary: 使用Adobe PDF Extract API从PDF文件中提取图片，接收Adobe API凭据，
#   调用Adobe服务提取图片并转换为base64格式，返回与HTML版本相同格式的结果
# intent: 实现PDF图片提取功能，支持Adobe免费套餐（每月500次），处理Adobe返回的ZIP文件并提取图片
async def handle_article_extract_images_from_pdf(data: ArticleExtractImagesReq):
    # 1. 验证Adobe API凭据
    if not data.pdf_services_client_id or not data.pdf_services_client_secret:
        return {"ok": False, "error": "Adobe PDF Services credentials required for PDF extraction"}
    
    try:
        # 2. 读取PDF文件
        abs_path = resolve_article_abs_path(data.article_id)
        with open(abs_path, 'rb') as f:
            input_stream = f.read()
        
        # 3. 初始化Adobe PDF Services
        credentials = ServicePrincipalCredentials(
            client_id=data.pdf_services_client_id,
            client_secret=data.pdf_services_client_secret
        )
        pdf_services = PDFServices(credentials=credentials)
        
        # 4. 上传PDF文件
        input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)
        
        # 5. 创建提取参数（只提取图片）
        extract_pdf_params = ExtractPDFParams(
            elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES],
            elements_to_extract_renditions=[ExtractRenditionsElementType.FIGURES],
        )
        
        # 6. 提交提取任务
        extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)
        location = pdf_services.submit(extract_pdf_job)
        pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)
        
        # 7. 获取结果ZIP文件，保存到PDF同目录以便调试和复用
        result_asset = pdf_services_response.get_result().get_resource()
        stream_asset = pdf_services.get_content(result_asset)
        zip_bytes = stream_asset.get_input_stream()
        
        # DOC-BEGIN id=handlers/articles/pdf-zip-save#1 type=behavior v=1
        # summary: 将Adobe返回的ZIP字节流保存到PDF文件同目录，文件名与PDF相同（.zip后缀）；
        #   已存在则跳过（避免重复写入），写入失败不阻塞主流程
        # intent: ZIP包含structuredData.json和所有图片，保存后可用于调试/复用/离线分析，
        #   避免每次重新调用Adobe API消耗配额
        zip_save_path = os.path.splitext(abs_path)[0] + ".adobe.zip"
        try:
            if not os.path.exists(zip_save_path):
                with open(zip_save_path, "wb") as zf:
                    zf.write(zip_bytes)
                logger.info(f"Saved Adobe ZIP to {zip_save_path}")
            else:
                logger.info(f"Adobe ZIP already exists at {zip_save_path}, skipping save")
        except Exception as e:
            logger.warning(f"Failed to save Adobe ZIP: {e}")
        # DOC-END id=handlers/articles/pdf-zip-save#1

        # 8. 处理ZIP文件并提取图片和文本，解压内容保存到 *_extracted 目录
        # DOC-BEGIN id=handlers/articles/pdf-output-dir#1 type=behavior v=1
        # summary: 计算解压输出目录为PDF文件名去掉扩展名后加_extracted后缀（如 xxx_extracted/），
        #   传入extract_images_from_adobe_zip使其将structuredData.json、figure图片、table图片落盘
        # intent: output_dir与PDF/ZIP同级存放，便于统一管理；目录创建失败时不阻塞主流程，
        #   仅跳过落盘（output_dir传None）
        pdf_stem = os.path.splitext(abs_path)[0]
        output_dir = pdf_stem + "_extracted"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create output dir {output_dir}: {e}")
            output_dir = None
        # DOC-END id=handlers/articles/pdf-output-dir#1
        zip_result = await extract_images_from_adobe_zip(zip_bytes, output_dir=output_dir)
        images_data = zip_result["images"]

        return {
            "ok": True,
            "article_id": data.article_id,
            "total": len(images_data),
            "images": images_data,
            "text": zip_result["text"],
        }

    except Exception as e:
        logging.exception(f'Adobe PDF Extract API error: {e}')
        return {"ok": False, "error": f"Adobe PDF Extract API failed: {str(e)}"}
# DOC-END id=handlers/articles/extract-images-from-pdf#1

# DOC-BEGIN id=handlers/articles/extract-images-from-adobe-zip#3 type=behavior v=3
# summary: 从Adobe PDF Extract API返回的ZIP文件中提取图片和正文文本。
#   图片来源：structuredData.json中所有Figure元素的filePaths（含Figure[2]等变体）+ tables/目录下的.png文件。
#   正文来源：structuredData.json中Path类型为P/H1/H2/Title/Footnote的Text字段拼接。
#   所有图片经过PIL获取实际尺寸，转为base64返回。
#   可选output_dir参数：指定后将structuredData.json和每张图片（figure_N.png/table_N.png）写入该目录，便于离线复用。
# intent: 原实现仅匹配路径精确以"/Figure"结尾的元素（仅1个），遗漏了Figure[2]-Figure[19]；
#   另外完全未提取正文文本，表格的xlsx也不发送给LLM（只发png图片）。
#   ZIP使用io.BytesIO内存流处理，同时可选地将内容持久化到磁盘。
async def extract_images_from_adobe_zip(zip_content: bytes, output_dir: str = None) -> dict:
    """
    从Adobe Extract API返回的ZIP中提取图片和正文文本。

    Parameters:
    zip_content (bytes): ZIP文件字节流
    output_dir (str, optional): 解压内容保存目录，如传入则将structuredData.json和图片落盘

    Returns:
    dict: {"images": list[dict], "text": str}
    """
    images_data = []
    full_text = ""

    try:
        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
            json_files = [f for f in zip_ref.namelist() if f.endswith('.json')]
            if not json_files:
                return {"images": [], "text": ""}

            # 读取 structuredData.json
            with zip_ref.open(json_files[0]) as json_file:
                structure_data = json.load(json_file)

            # DOC-BEGIN id=handlers/articles/adobe-zip-save-json#1 type=behavior v=1
            # summary: 将structuredData.json保存到output_dir（如果指定），便于离线分析和调试
            # intent: structuredData.json包含Adobe提取的所有元素信息（Path/Text/filePaths等），
            #   保存后可以不重新调用API而复用这些数据
            if output_dir:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    json_save_path = os.path.join(output_dir, "structuredData.json")
                    with open(json_save_path, "w", encoding="utf-8") as jf:
                        json.dump(structure_data, jf, ensure_ascii=False, indent=2)
                    logger.info(f"Saved structuredData.json to {json_save_path}")
                except Exception as e:
                    logger.warning(f"Failed to save structuredData.json: {e}")
            # DOC-END id=handlers/articles/adobe-zip-save-json#1

            elements = structure_data.get('elements', [])

            # 1. 提取正文文本：P/H1/H2/Title/Footnote（排除L参考文献）
            text_parts = []
            text_types = {"P", "H1", "H2", "Title", "Footnote"}
            for el in elements:
                path = el.get("Path", "")
                parts = [p for p in path.split("/") if p]
                if len(parts) >= 2:
                    base_cat = parts[1].split("[")[0]
                    if base_cat in text_types and "Text" in el:
                        text_parts.append(el["Text"])
            full_text = "\n".join(text_parts)

            # 2. 提取Figure图片：匹配 //Document/Figure 和 //Document/Figure[N]
            figure_pattern = re.compile(r"^//Document/Figure(\[\d+\])?$")
            count = 1

            for el in elements:
                path = el.get("Path", "")
                if not figure_pattern.match(path):
                    continue

                file_paths = el.get("filePaths", [])
                if not file_paths:
                    continue

                img_rel_path = file_paths[0]
                if img_rel_path not in zip_ref.namelist():
                    continue

                with zip_ref.open(img_rel_path) as img_file:
                    img_bytes = img_file.read()

                ext = img_rel_path.rsplit(".", 1)[-1].lower()
                if ext in ("jpg", "jpeg"):
                    mime_type = "image/jpeg"
                    pil_format = "JPEG"
                elif ext == "png":
                    mime_type = "image/png"
                    pil_format = "PNG"
                else:
                    continue

                # 获取实际尺寸
                orig_w, orig_h = Image.open(io.BytesIO(img_bytes)).size
                b64_content = base64.b64encode(img_bytes).decode("utf-8")

                # DOC-BEGIN id=handlers/articles/adobe-zip-save-figure#1 type=behavior v=1
                # summary: 将Figure图片保存到output_dir，文件名如figure_1.png/figure_2.jpg；
                #   需要重新从img_bytes打开流写入（原zip_ref.open流已在上文关闭）
                # intent: 图片落盘后可以独立于ZIP文件使用，便于调试和离线分析；
                #   文件名与images_data中的filename字段保持一致，便于对照
                if output_dir:
                    try:
                        save_path = os.path.join(output_dir, f"figure_{count}.{ext}")
                        with open(save_path, "wb") as sf:
                            sf.write(img_bytes)
                        logger.info(f"Saved figure image to {save_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save figure image {count}: {e}")
                # DOC-END id=handlers/articles/adobe-zip-save-figure#1

                images_data.append({
                    "index": count,
                    "filename": f"figure_{count}.{ext}",
                    "caption": f"Figure {count}",
                    "width": orig_w,
                    "height": orig_h,
                    "mime_type": mime_type,
                    "base64_content": f"data:{mime_type};base64,{b64_content}",
                })
                count += 1

            # 3. 提取表格图片：tables/*.png（跳过.xlsx）
            for entry in sorted(zip_ref.namelist()):
                if not entry.startswith("tables/"):
                    continue
                ext = entry.rsplit(".", 1)[-1].lower()
                if ext != "png":
                    continue

                with zip_ref.open(entry) as img_file:
                    img_bytes = img_file.read()

                tbl_img = Image.open(io.BytesIO(img_bytes))
                new_w, new_h = tbl_img.size
                b64_content = base64.b64encode(img_bytes).decode("utf-8")

                # DOC-BEGIN id=handlers/articles/adobe-zip-save-table#1 type=behavior v=1
                # summary: 将Table图片保存到output_dir，文件名如table_1.png/table_2.png
                # intent: 表格截图落盘，文件名与images_data中的filename字段保持一致；
                #   与Figure图片并列存储在同一个_extracted目录下
                if output_dir:
                    try:
                        tbl_basename = os.path.basename(entry)
                        tbl_count = tbl_basename.replace("table_", "").replace(".png", "")
                        save_name = f"table_{count}.png"
                        save_path = os.path.join(output_dir, save_name)
                        with open(save_path, "wb") as sf:
                            sf.write(img_bytes)
                        logger.info(f"Saved table image to {save_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save table image: {e}")
                # DOC-END id=handlers/articles/adobe-zip-save-table#1

                images_data.append({
                    "index": count,
                    "filename": f"table_{count}.png",
                    "caption": f"Table {count}",
                    "width": new_w,
                    "height": new_h,
                    "mime_type": "image/png",
                    "base64_content": f"data:image/png;base64,{b64_content}",
                })
                count += 1

    except Exception as e:
        logger.error(f"Error processing Adobe ZIP: {e}")
        logger.exception(e)

    return {
        "images": images_data,
        "text": full_text,
    }
# DOC-END id=handlers/articles/extract-images-from-adobe-zip#3
# DOC-END id=handlers/articles/extract-images-from-adobe-zip#2


async def handle_article_list(data: ArticleListReq):
    return {"ok": True, "tree": build_articles_tree(ARTICLES_ROOT)}

async def handle_task_select_article(data: TaskSelectArticleReq):
    abs_path = resolve_article_abs_path(data.article_id)
    title, raw_html = load_raw_html(abs_path)
    cleaned = clean_html_for_injection(raw_html)
    await inject_article_to_task(
        task_id=data.task_id,
        article_id=data.article_id,
        title=title,
        content=cleaned,
    )
    return {"ok": True, "msg": f"Article '{title}' injected successfully."}

async def handle_article_generate_blog(data: ArticleGenerateBlogReq):
    """
    状态机：
    - completed : blog 已生成，直接返回内容
    - running   : 正在生成（.lock 存在）
    - queued    : 已入队等待（.queued 存在）
    - failed    : 生成失败（.error 存在）
    - queued    : 本次刚入队（ACK）
    """

    # --------------------------------------------------
    # 0. Resolve paths
    # --------------------------------------------------
    abs_path   = resolve_article_abs_path(data.article_id)
    txt_path   = article_txt_path_from_html(abs_path)
    lock_path  = article_lock_path(abs_path)
    err_path   = article_error_path(abs_path)
    queue_path = article_queued_path(abs_path)
    all_img_path = abs_path + ".images.json"

    # --------------------------------------------------
    # 1. COMPLETED
    # --------------------------------------------------
    if os.path.exists(txt_path):
        blog_md = read_text_file(txt_path)

        img_result = await handle_article_extract_images(
            ArticleExtractImagesReq(article_id=data.article_id)
        )
        all_images = img_result.get("images", [])
        write_text_file(all_img_path, json.dumps(all_images, ensure_ascii=False, indent=2))
        images = json.loads(read_text_file(all_img_path))

        return {
            "ok": True,
            "status": "completed",
            "blog_markdown": blog_md,
            "images": images,
        }

    # --------------------------------------------------
    # 2. RUNNING
    # --------------------------------------------------
    if os.path.exists(lock_path):
        return {"ok": True, "status": "running"}

    # --------------------------------------------------
    # 3. QUEUED
    # --------------------------------------------------
    if os.path.exists(queue_path):
        return {"ok": True, "status": "queued"}

    # --------------------------------------------------
    # 4. FAILED
    # --------------------------------------------------
    if os.path.exists(err_path):
        return {
            "ok": False,
            "status": "failed",
            "error": read_text_file(err_path),
        }

    # --------------------------------------------------
    # 5. 写 .queued 文件，提交到全局 FIFO 队列
    # --------------------------------------------------
    # DOC-BEGIN id=handlers/bloggen/enqueue#1 type=behavior v=1
    # summary: 写入 .queued 标记文件后将 worker 提交到 BLOG_JOB_QUEUE；
    #   worker 启动时删除 .queued 并写入 .lock，结束时删除 .lock；
    #   串行化由 BLOG_JOB_QUEUE 的单 worker 线程保证，无需额外全局锁。
    # intent: .queued 文件让前端/后续请求感知"已入队但未开始"状态，
    #   避免同一篇文章重复入队；BLOG_JOB_QUEUE 是全局 FIFO，保证跨文章严格串行。
    write_text_file(queue_path, json.dumps({
        "article_id": data.article_id,
        "task_id": data.task_id,
    }, ensure_ascii=False))

    def worker():
        logger.info(f"[Worker] Entered worker for article_id={data.article_id}")
        # 从 queued → running
        try:
            os.remove(queue_path)
            logger.info(f"[Worker] Removed .queued file")
        except FileNotFoundError:
            logger.warning(f"[Worker] .queued file already gone")

        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            logger.info(f"[Worker] Created .lock file")
        except FileExistsError:
            logger.warning(f"[Worker] .lock already exists, aborting")
            return  # 已有其他进程在跑，放弃

        try:
            # PDF 文件不应调用 load_raw_html（会读取二进制乱码），
            # 直接从文件名取标题；HTML 文件正常解析标题和正文
            is_pdf = abs_path.lower().endswith('.pdf')
            if is_pdf:
                title = re.sub(r'\.pdf$', '', os.path.basename(abs_path), flags=re.IGNORECASE) or "Untitled"
                cleaned_text = ""
                logger.info(f"[Worker] Start generating blog for PDF: {title}")
            else:
                title, raw_html = load_raw_html(abs_path)
                cleaned_text = clean_html_for_injection(raw_html)
                logger.info(f"[Worker] Start generating blog for: {title}")

            # DOC-BEGIN id=handlers/bloggen/extract-all#1 type=behavior v=1
            # summary: 提取文章图片和正文文本；PDF路径通过extract_images_from_adobe_zip同时提取text字段（Adobe结构化提取），
            #   HTML路径extract_images返回的images已有base64_content，text为空由cleaned_text兜底。
            #   images_b64列表包含所有图片的base64字符串，传给多模态LLM直接"看图"。
            # intent: 聚合两个来源的数据——img_result["images"]含base64/img_meta，img_result["text"]含PDF正文；
            #   HTML文本仍用cleaned_text（从clean_html_for_injection来的），PDF文本优先用extract的text。
            logger.info(f"[BlogGen][{data.article_id}] Extracting images and text...")
            # DOC-BEGIN id=handlers/bloggen/extract-with-credentials#1 type=behavior v=2
            # summary: 提取文章图片和正文，传递前端传入的Adobe PDF Services凭据；
            #   PDF文件需要凭据调用Adobe API，HTML文件忽略凭据直接解析。
            #   增加 ok 检查：如果图片提取返回 ok=False（如Adobe超时），立即抛异常终止 worker，
            #   避免用空内容继续生成无意义的blog。
            # intent: 前端在用户选择PDF生成blog时传入凭据，后端透传到图片提取函数；
            #   凭据为None时HTML路径不受影响。
            #   之前缺少 ok 检查，导致 Adobe 超时/认证失败后仍继续生成空blog。
            img_result = asyncio.run(
                handle_article_extract_images(
                    ArticleExtractImagesReq(
                        article_id=data.article_id,
                        pdf_services_client_id=data.pdf_services_client_id,
                        pdf_services_client_secret=data.pdf_services_client_secret,
                    )
                )
            )
            # DOC-END id=handlers/bloggen/extract-with-credentials#1

            # DOC-BEGIN id=handlers/bloggen/extract-result-check#1 type=behavior v=1
            # summary: 检查图片提取结果的 ok 字段；若为 False 则抛出异常使 worker 进入错误处理流程，
            #   写入 .error 文件供前端展示，而非用空 images/空 text 继续生成。
            # intent: PDF图片提取（Adobe API）可能因网络超时、凭据错误、文件过大等原因失败；
            #   之前失败后返回 ok=False 但未被检查，导致后续 generate_blog_from_article_tree
            #   收到 0 images + 0 text，生成无意义输出还消耗 LLM token。
            if not img_result.get("ok", False):
                error_msg = img_result.get("error", "Image extraction failed")
                raise RuntimeError(f"Image extraction failed: {error_msg}")
            # DOC-END id=handlers/bloggen/extract-result-check#1

            all_images = img_result.get("images", [])
            write_text_file(all_img_path, json.dumps(all_images, ensure_ascii=False, indent=2))
            logger.info(f"[BlogGen][{data.article_id}] Extracted {len(all_images)} images")

            # PDF提取的文本优先使用（更准确），HTML用cleaned_text
            extracted_text = img_result.get("text", "")
            article_text_for_gen = extracted_text if extracted_text else cleaned_text
            logger.info(f"[BlogGen][{data.article_id}] Text length: {len(article_text_for_gen)} chars")

            # DOC-BEGIN id=handlers/bloggen/image-preprocessing#1 type=behavior v=1
            # summary: 收集所有图片的base64并进行分辨率预处理，降低LLM token消耗；
            #   使用LLM.reduce_image_resolution按PDF_IMAGE_SCALE_FACTOR缩放；
            #   仅对大于阈值（800px宽或高）的图片进行缩放，小图保持原样
            # intent: 高分辨率图片直接发送给多模态LLM会消耗大量token（按像素计费）；
            #   预处理可减少50-75%的token消耗，同时保持足够清晰度供LLM理解内容
            images_b64 = []
            from src.services.llm import LLM
            for img in all_images:
                b64 = img.get("base64_content", "")
                if not b64:
                    continue
                # 对大图进行缩放预处理
                w = img.get("width", 0) or 0
                h = img.get("height", 0) or 0
                if w > 800 or h > 800:
                    try:
                        b64 = LLM.reduce_image_resolution(b64, scale_factor=0.5)
                        logger.info(f"[BlogGen][{data.article_id}] Resized image {img.get('index')}: {w}x{h} -> reduced")
                    except Exception as e:
                        logger.warning(f"[BlogGen][{data.article_id}] Failed to resize image {img.get('index')}: {e}")
                images_b64.append(b64)
            logger.info(f"[BlogGen][{data.article_id}] Prepared {len(images_b64)} images for multimodal (after preprocessing)")
            # DOC-END id=handlers/bloggen/image-preprocessing#1

            image_catalog = [
                {
                    "fig": str(img["index"]),
                    "caption": img["caption"],
                    "width": img["width"],
                    "height": img["height"],
                }
                for img in all_images
            ]

            config = BlogGenConfig(
                model_name=data.model_name,
                api_key=data.api_key,
                llm_url=data.llm_url,
                style=data.style or "math",
                l1_points=5,
                l2_points=4,
            )

            task_dir = f"{DATA_DIR}/tasks/{data.task_id}"
            tc = Tool_Calls(LOG_DIR=task_dir, MAX_CHAR=800000, mode="Summary")

            logger.info(f"[BlogGen][{data.article_id}] Starting blog generation with multimodal LLM...")
            result = generate_blog_from_article_tree(
                task_id=data.task_id,
                article_id=data.article_id,
                article_title=title,
                article_text=article_text_for_gen,
                image_catalog=image_catalog,
                images_b64=images_b64,
                config=config,
                tc=tc,
            )
            # DOC-END id=handlers/bloggen/extract-all#1

            blog_md = result["blog_markdown"]
            write_text_file(txt_path, blog_md)
            logger.info(f"Complete generating blog for {title}.")

        except Exception as e:
            # DOC-BEGIN id=handlers/bloggen/worker-error-logging#1 type=behavior v=1
            # summary: 捕获 worker 异常后通过 traceback + logger.error 输出到主线程 stderr，
            #   同时写入 .error 文件供前端状态机查询
            # intent: 使用 python -m server 运行时，logger.error 和 traceback.print_exc 都会
            #   输出到终端，方便开发者实时看到生成失败原因；.error 文件是持久化状态，
            #   供后续请求返回 failed 状态
            traceback.print_exc()
            logger.error(f"Blog generation failed for article_id={data.article_id}: {e}")
            write_text_file(err_path, str(e))
            # DOC-END id=handlers/bloggen/worker-error-logging#1

        finally:
            # DOC-BEGIN id=handlers/bloggen/finally-cleanup#1 type=behavior v=1
            # summary: 清理 .lock 文件，并在 .txt 和 .error 都不存在时写入兜底 .error 文件
            # intent: 如果生成过程中异常被捕获但 write_text_file(err_path) 也失败（如磁盘满），
            #   会导致四个标记文件全部不存在，下次请求走入 step 5 重新入队，造成无限重试循环；
            #   兜底写入确保状态机不会回到 "none"
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass
            if not os.path.exists(txt_path) and not os.path.exists(err_path):
                try:
                    write_text_file(err_path, "Unknown error: blog generation failed without details")
                except Exception:
                    pass
            # DOC-END id=handlers/bloggen/finally-cleanup#1

    BLOG_JOB_QUEUE.submit(worker)

    return {"ok": True, "status": "queued"}
