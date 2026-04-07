
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
# DOC-BEGIN id=handlers/articles/blog-generator-imports#1 type=dependency v=1
# summary: 从博客生成器模块导入所需函数和类，包括JSON解析函数_extract_json_array
# intent: _extract_json_array函数在Figure Index解析中使用，需要从article_blog_generator模块导入；
#   其他函数generate_blog_from_article_tree、BlogGenConfig、_new_llm在博客生成流程中使用
# DOC-BEGIN id=handlers/articles/blog-generator-imports#1 type=dependency v=2
# summary: 从博客生成器模块导入所需函数和类，包括JSON解析函数_extract_json_array；
#   同时定义Figure Index批量生成相关常量
# intent: _extract_json_array函数在Figure Index解析中使用，需要从article_blog_generator模块导入；
#   FIGURE_BATCH_SIZE控制每次发送给LLM的图片数量，增大可减少API调用次数但增加单次token消耗
from src.utils.article_blog_generator import generate_blog_from_article_tree, BlogGenConfig, _new_llm, _extract_json_array

# Figure Index 批量生成逻辑：总图片平均分为3批处理
# DOC-END id=handlers/articles/blog-generator-imports#1
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

        # 和PDF逻辑统一：过滤保留最大25张，还原原始顺序，重编号
        valid_images = images_data.copy()
        MAX_IMAGES = 25
        original_total = len(valid_images)
        logger.info(f"[HTML Extract] Filtered images from {original_total} to {min(original_total, MAX_IMAGES)}, kept top {min(original_total, MAX_IMAGES)} large valid images in original article order")
        
        if len(valid_images) > MAX_IMAGES:
            # 按面积降序排序选前25张大图
            valid_images.sort(key=lambda x: x["width"]*x["height"], reverse=True)
            valid_images = valid_images[:MAX_IMAGES]
            # 还原原始文章的出现顺序
            valid_images.sort(key=lambda x: x["index"])
        
        # 重编号，确保index连续不重复
        for new_idx, img in enumerate(valid_images, 1):
            ext = img["filename"].split(".")[-1]
            img["index"] = new_idx
            img["filename"] = f"figure_{new_idx}.{ext}"
            img["caption"] = f"Figure {new_idx}"
        
        images_data = valid_images
    
    # HTML和PDF返回格式完全统一
    return {"ok": True, "article_id": data.article_id, "total": len(images_data), "images": images_data}
# DOC-END id=handlers/articles/extract-images#1

# DOC-BEGIN id=handlers/articles/extract-images-from-pdf#1 type=api v=1
# summary: 使用Adobe PDF Extract API从PDF文件中提取图片，接收Adobe API凭据，
#   调用Adobe服务提取图片并转换为base64格式，返回与HTML版本相同格式的结果
# intent: 实现PDF图片提取功能，支持Adobe免费套餐（每月500次），处理Adobe返回的ZIP文件并提取图片
async def handle_article_extract_images_from_pdf(data: ArticleExtractImagesReq):
    abs_path = resolve_article_abs_path(data.article_id)
    pdf_stem = os.path.splitext(abs_path)[0]
    zip_path = pdf_stem + ".adobe.zip"

    # DOC-BEGIN id=handlers/articles/pdf-zip-reuse#1 type=behavior v=1
    # summary: 检查PDF同目录下是否存在之前保存的 .adobe.zip 文件；
    #   如果存在，直接复用ZIP内容，跳过Adobe API调用，节省配额和时间。
    #   ZIP不存在时才需要Adobe凭据去调用API生成。
    # intent: Adobe API有每月500次免费额度限制，复用ZIP可以避免重复消耗；
    #   用户重新生成blog时，图片提取结果不变，无需重新调用Adobe。
    if os.path.exists(zip_path):
        logger.info(f"Found existing Adobe ZIP at {zip_path}, reusing...")
        try:
            with open(zip_path, "rb") as zf:
                zip_bytes = zf.read()

            # 确定输出目录
            output_dir = pdf_stem + "_extracted"
            
            # 清理output_dir，确保目录内只有本次提取的内容
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

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
            logger.warning(f"Failed to reuse ZIP {zip_path}: {e}, will call Adobe API")
    # DOC-END id=handlers/articles/pdf-zip-reuse#1

    # ZIP不存在，需要调用Adobe API
    # 1. 验证Adobe API凭据
    if not data.pdf_services_client_id or not data.pdf_services_client_secret:
        return {"ok": False, "error": "Adobe PDF Services credentials required for PDF extraction (no cached ZIP found)"}
    
    try:
        # 2. 读取PDF文件
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
        
        # 5. 创建提取参数：提取文本+表格结构，同时获取Figure和Table的图片rendition
        extract_pdf_params = ExtractPDFParams(
            elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES],
            elements_to_extract_renditions=[ExtractRenditionsElementType.FIGURES, ExtractRenditionsElementType.TABLES],
        )
        
        # 6. 提交提取任务
        extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)
        location = pdf_services.submit(extract_pdf_job)
        pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)
        
        # 7. 获取结果ZIP文件，保存到PDF同目录以便复用
        result_asset = pdf_services_response.get_result().get_resource()
        stream_asset = pdf_services.get_content(result_asset)
        zip_bytes = stream_asset.get_input_stream()

        # 保存ZIP
        try:
            with open(zip_path, "wb") as zf:
                zf.write(zip_bytes)
            logger.info(f"Saved Adobe ZIP to {zip_path}")
        except Exception as e:
            logger.warning(f"Failed to save Adobe ZIP: {e}")

        # 8. 解压并提取图片
        output_dir = pdf_stem + "_extracted"
        # DOC-BEGIN id=handlers/articles/clean-output-dir#2 type=behavior v=2
        # summary: 在解压ZIP前先清理output_dir，确保目录内只有本次提取的内容。
        #   保留structuredData.json和figindex.json（如果存在），删除其他所有文件。
        # intent: 严格满足"extracted文件夹内部只有保存好的图片和两个json文件"的要求，
        #   防止之前提取的图片或临时文件污染当前结果。
        if os.path.exists(output_dir):
            for fname in os.listdir(output_dir):
                if fname in ["structuredData.json"] or fname.endswith(".figindex.json"):
                    continue
                fpath = os.path.join(output_dir, fname)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                    elif os.path.isdir(fpath):
                        import shutil
                        shutil.rmtree(fpath)
                except Exception as e:
                    logger.warning(f"Failed to cleanup {fpath}: {e}")
        else:
            os.makedirs(output_dir, exist_ok=True)
        # DOC-END id=handlers/articles/clean-output-dir#2

        # 传递output_dir参数，确保图片落盘保存
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

# DOC-BEGIN id=handlers/articles/generate-figure-index-batch#1 type=function v=1
# summary: 为一批图片生成结构化Figure Index注释。
#   接收当前批次的图片数据、文章标题、原文文本、博客内容和原始编号列表；
#   内部使用LLM生成1,2,...,len(batch)的临时编号，然后映射回原始编号。
# intent: 将Figure Index生成拆分为多批次处理，避免单次LLM调用发送过多图片导致token超限；
#   批次间完全独立，只需原文+博客上下文，不依赖前序批次结果。
def generate_figure_index_batch(
    images_batch: list,
    article_title: str,
    article_text: str,
    blog_markdown: str,
    batch_original_nums: list,
    config: BlogGenConfig,
) -> list:
    """
    为一批图片生成结构化区域注释。

    Parameters:
    images_batch (list): 当前批次的图片数据列表，每项含base64_content、index等
    article_title (str): 文章标题
    article_text (str): 原始文章文本（可能已截断）
    blog_markdown (str): 已生成的博客内容
    batch_original_nums (list): 当前批次图片的原始编号列表，顺序与images_batch一致
    config (BlogGenConfig): LLM配置

    Returns:
    list: 结构化的Figure Index列表，fig_num已映射回原始编号

    Raises:
    RuntimeError: LLM调用失败或JSON解析失败时抛出异常
    """
    batch_size = len(images_batch)
    if batch_size == 0:
        return []

    # 准备传给多模态LLM的图片base64列表
    fig_images_b64 = [img["base64_content"] for img in images_batch]

    # 构造图片列表描述（使用临时编号1,2,...,N）
    fig_list_desc = "\n".join([
        f"FIG {i+1} 对应博客中的[[FIG:{batch_original_nums[i]}]]"
        for i in range(batch_size)
    ])

    # 构造提示词
    batch_prompt = f"""
文章标题：{article_title}
原始文章正文：{article_text}
已生成的博客主体内容：{blog_markdown}

以下是博客中用到的部分图片（共{batch_size}张），请按列表顺序处理：
{fig_list_desc}

请为每张图片生成结构化的区域注释。

输出要求：严格返回JSON数组，不要输出任何其他内容，格式如下：
[
  {{
    "fig_num": 图片编号（从1开始，对应上面的FIG 1, FIG 2, ...）,
    "regions": [
      {{"bbox": [x1,y1,x2,y2], "text": "该区域的详细说明文字"}}
    ]
  }}
]
要求：
1. bbox是图片内的归一化相对坐标，范围0到1，格式为[左上角x, 左上角y, 右下角x, 右下角y]
2. 为所有图片的所有关键元素（子图、坐标轴、数据曲线、表格核心单元格等）都生成独立的区域注释
3. 所有描述文本为中文，完全贴合文章和博客的业务内容
4. 严格符合JSON格式，不要添加任何额外解释、标记或非JSON内容
5. 必须为每张图片都生成注释，fig_num从1到{batch_size}
"""

    # 调用多模态LLM生成结构化索引
    llm_figindex = _new_llm(config)
    figindex_raw = llm_figindex.query_multimodal(batch_prompt, fig_images_b64, verbose=True)

    # 智能提取JSON部分
    def _smart_extract_json(text: str) -> str:
        """从文本中智能提取JSON数组：找第一个[和最后一个]之间的内容"""
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and start < end:
            return text[start:end+1]
        return text

    json_text = _smart_extract_json(figindex_raw)

    try:
        parsed_index = _extract_json_array(json_text)
    except Exception as e:
        logger.error(f"[FigureIndexBatch] Failed to parse JSON: {e}")
        logger.error(f"[FigureIndexBatch] Raw output (first 500 chars): {figindex_raw[:500]}")
        raise RuntimeError(f"Figure Index batch JSON parsing failed: {e}")

    # 将临时编号映射回原始编号
    for i, entry in enumerate(parsed_index):
        if i < len(batch_original_nums):
            original_num = batch_original_nums[i]
            entry["fig_num"] = original_num
            entry["original_image_index"] = original_num

    logger.info(f"[FigureIndexBatch] Generated index for {len(parsed_index)} figures, mapped to originals: {batch_original_nums}")
    return parsed_index
# DOC-END id=handlers/articles/generate-figure-index-batch#1

# DOC-BEGIN id=handlers/articles/extract-images-from-adobe-zip#4 type=behavior v=4
# summary: 从Adobe PDF Extract API返回的ZIP文件中提取图片和正文文本。
#   图片来源：直接遍历ZIP中 figures/ 和 tables/*.png 目录，不需要解析JSON找路径。
#   正文来源：structuredData.json中Path类型为P/H1/H2/Title/Footnote的Text字段拼接。
#   所有图片经过PIL获取实际尺寸，转为base64返回。
#   可选output_dir参数：指定后将structuredData.json和每张图片落盘，便于离线复用。
# intent: Adobe提取结果中figures/目录直接包含所有Figure图片，tables/目录包含Table截图和xlsx；
#   通过JSON的filePaths找图片路径是多余且容易出错的，直接遍历目录更简单可靠。
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
            # === 1. 提取正文文本 ===
            json_files = [f for f in zip_ref.namelist() if f.endswith('.json')]
            if json_files:
                with zip_ref.open(json_files[0]) as json_file:
                    structure_data = json.load(json_file)

                # 保存structuredData.json到output_dir
                if output_dir:
                    try:
                        os.makedirs(output_dir, exist_ok=True)
                        json_save_path = os.path.join(output_dir, "structuredData.json")
                        with open(json_save_path, "w", encoding="utf-8") as jf:
                            json.dump(structure_data, jf, ensure_ascii=False, indent=2)
                        logger.info(f"Saved structuredData.json to {json_save_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save structuredData.json: {e}")

                # 提取文本：P/H1/H2/Title/Footnote
                elements = structure_data.get('elements', [])
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

            # === 2. 提取图片：直接遍历 figures/ 和 tables/*.png ===
            # DOC-BEGIN id=handlers/articles/adobe-zip-extract-figures#1 type=behavior v=1
            # summary: 遍历ZIP中 figures/ 目录下的所有图片文件（png/jpg/jpeg），
            #   按文件名排序后依次读取，获取尺寸并转为base64。
            # intent: figures/ 目录是Adobe提取结果的标准结构，直接包含所有Figure的图片文件，
            #   不需要通过JSON的filePaths间接查找，更简单可靠。
            count = 1
            all_entries = sorted(zip_ref.namelist())

            for entry in all_entries:
                if not entry.startswith("figures/"):
                    continue
                # 跳过目录本身
                if entry.endswith("/"):
                    continue
                ext = entry.rsplit(".", 1)[-1].lower() if "." in entry else ""
                if ext not in ("png", "jpg", "jpeg"):
                    continue

                with zip_ref.open(entry) as img_file:
                    img_bytes = img_file.read()

                if ext in ("jpg", "jpeg"):
                    mime_type = "image/jpeg"
                else:
                    mime_type = "image/png"

                orig_w, orig_h = Image.open(io.BytesIO(img_bytes)).size
                b64_content = base64.b64encode(img_bytes).decode("utf-8")

                # 保存到output_dir
                if output_dir:
                    try:
                        save_name = f"figure_{count}.{ext}"
                        save_path = os.path.join(output_dir, save_name)
                        with open(save_path, "wb") as sf:
                            sf.write(img_bytes)
                    except Exception as e:
                        logger.warning(f"Failed to save figure image {count}: {e}")

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
            # DOC-END id=handlers/articles/adobe-zip-extract-figures#1

            
# 3. 提取表格图片：tables/*.png（跳过.xlsx等非图片文件）
            # DOC-BEGIN id=handlers/articles/adobe-zip-extract-tables#1 type=behavior v=2
            # summary: 遍历ZIP中 tables/ 目录，仅提取.png图片文件，跳过.xlsx等非图片；
            #   暂不保存到文件系统，等过滤后再统一保存。
            # intent: tables/ 目录可能包含xlsx（原始数据）和png（截图），只需png用于LLM理解。
            for entry in all_entries:
                if not entry.startswith("tables/"):
                    continue
                ext = entry.rsplit(".", 1)[-1].lower() if "." in entry else ""
                if ext != "png":
                    continue

                with zip_ref.open(entry) as img_file:
                    img_bytes = img_file.read()

                tbl_w, tbl_h = Image.open(io.BytesIO(img_bytes)).size
                b64_content = base64.b64encode(img_bytes).decode("utf-8")

                # 注意：这里不保存到output_dir，等过滤后再保存
                images_data.append({
                    "index": count,
                    "filename": f"table_{count}.png",
                    "caption": f"Table {count}",
                    "width": tbl_w,
                    "height": tbl_h,
                    "mime_type": "image/png",
                    "base64_content": f"data:image/png;base64,{b64_content}",
                })
                count += 1
            # DOC-END id=handlers/articles/adobe-zip-extract-tables#1

            
# DOC-BEGIN id=handlers/articles/adobe-zip-filter-images#1 type=behavior v=3
            # summary: 过滤图片并保存到文件系统：按面积排序保留最大的25张，还原原文章出现顺序，重编号后统一保存到output_dir
            # intent: 先在内存中完成过滤和重编号，然后一次性清理output_dir并保存，确保目录内只有最终保留的图片
            valid_images = images_data.copy()
            MAX_IMAGES = 25
            original_total = len(valid_images)
            logger.info(f"[Adobe ZIP] Filtered images from {original_total} to {min(original_total, MAX_IMAGES)}, kept top {min(original_total, MAX_IMAGES)} large valid images in original article order")
            
            if len(valid_images) > MAX_IMAGES:
                # 按面积降序排序选前25张大图
                valid_images.sort(key=lambda x: x["width"]*x["height"], reverse=True)
                valid_images = valid_images[:MAX_IMAGES]
                # 还原原始文章的出现顺序
                valid_images.sort(key=lambda x: x["index"])
            
            # 重编号，确保index连续不重复
            for new_idx, img in enumerate(valid_images, 1):
                ext = img["filename"].split(".")[-1]
                img["index"] = new_idx
                img["filename"] = f"figure_{new_idx}.{ext}"
                img["caption"] = f"Figure {new_idx}"
            
            images_data = valid_images
            
            # 清理output_dir并保存过滤后的图片
            if output_dir:
                # 清理所有旧文件，只保留structuredData.json
                for fname in os.listdir(output_dir):
                    if fname == "structuredData.json":
                        continue
                    fpath = os.path.join(output_dir, fname)
                    try:
                        if os.path.isfile(fpath):
                            os.remove(fpath)
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {fpath}: {e}")
                
                # 保存过滤后的图片
                for img in images_data:
                    try:
                        b64_data = img["base64_content"].split(",", 1)[1] if "," in img["base64_content"] else img["base64_content"]
                        img_bytes = base64.b64decode(b64_data)
                        save_path = os.path.join(output_dir, img["filename"])
                        with open(save_path, "wb") as f:
                            f.write(img_bytes)
                    except Exception as e:
                        logger.warning(f"Failed to save image {img['filename']}: {e}")
            # DOC-END id=handlers/articles/adobe-zip-filter-images#1

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

        # 从 .images.json 读取预处理后的图片（PDF和HTML统一）
        all_images = []
        if os.path.exists(all_img_path):
            all_images = json.loads(read_text_file(all_img_path))
            logger.info(f"[COMPLETED] Loaded {len(all_images)} images from .images.json")
        else:
            logger.warning(f"[COMPLETED] No images.json found for {data.article_id}")

        # DOC-BEGIN id=handlers/bloggen/frontend-filter-renumber#1 type=behavior v=1
        # summary: 返回前端前做图片过滤和重编号：
        #   1. 从blog中提取实际引用的图片编号
        #   2. 只保留被引用的图片
        #   3. 重新编号使FIG编号连续（1,2,3...）
        #   4. 同步更新blog中的FIG占位符
        # intent: 过滤和重编号仅在返回前端时执行，不保存到文件系统；
        #   文件系统保存原始编号的blog和images，便于重新生成时保持一致性
        used_fig_nums = set()
        for match in re.findall(r'\[\[FIG:(\d+)\]\]', blog_md):
            used_fig_nums.add(int(match))
        logger.info(f"[COMPLETED] Blog references figures: {sorted(used_fig_nums)}")

        # 筛选被引用的图片，保持原始顺序
        filtered_images = [img for img in all_images if img["index"] in used_fig_nums]

        # 图片编号和博客引用已在生成时永久固定，无需再次修改
        logger.info(f"[COMPLETED] Filtered: {len(all_images)} -> {len(filtered_images)} images (numbers already fixed)")
        # DOC-END id=handlers/bloggen/frontend-filter-renumber#1

        if filtered_images:
            first_b64_len = len(filtered_images[0].get("base64_content", ""))
            logger.info(f"[COMPLETED] Returning {len(filtered_images)} images to frontend, first base64 length: {first_b64_len}")

        # DOC-BEGIN id=handlers/bloggen/completed-response#1 type=behavior v=1
        # summary: 构建已完成状态的响应，包含博客内容、过滤后的图片和Figure Index
        #   Figure Index从extracted文件夹中的.figindex.json文件加载，如果文件不存在则为None
        # intent: 前端需要Figure Index数据用于展示图片区域注释，从文件系统读取已保存的Figure Index；
        #   文件可能不存在（如没有图片或生成失败），此时返回None表示无Figure Index
        figure_index = None
        if os.path.exists(abs_path):
            # 构建Figure Index文件路径
            article_stem = os.path.splitext(abs_path)[0]
            output_dir = article_stem + "_extracted"
            pdf_basename = os.path.splitext(os.path.basename(abs_path))[0]
            figindex_path = os.path.join(output_dir, f"{pdf_basename}.figindex.json")
            
            if os.path.exists(figindex_path):
                try:
                    with open(figindex_path, "r", encoding="utf-8") as f:
                        figure_index = json.load(f)
                    logger.info(f"[COMPLETED] Loaded Figure Index with {len(figure_index)} entries from {figindex_path}")
                except Exception as e:
                    logger.error(f"[COMPLETED] Failed to load Figure Index from {figindex_path}: {e}")
                    figure_index = None
        
        return {
            "ok": True,
            "status": "completed",
            "blog_markdown": blog_md,
            "images": filtered_images,
            "figure_index": figure_index,  # 新增字段：结构化的Figure Index数据
        }
        # DOC-END id=handlers/bloggen/completed-response#1

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
            # DOC-BEGIN id=handlers/bloggen/async-safe#1 type=behavior v=1
            # summary: 安全调用 async 函数 handle_article_extract_images，兼容已在运行的事件循环。
            #   若当前线程已有事件循环（如 FastAPI/Starlette），asyncio.run() 会抛 RuntimeError，
            #   此时改用 loop.run_until_complete() 在已有循环上执行。
            # intent: worker 在 BLOG_JOB_QUEUE 线程中运行，通常无事件循环，asyncio.run() 即可；
            #   但某些部署方式（如嵌套在 async handler 中 submit）可能导致已有循环存在。
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            extract_req = ArticleExtractImagesReq(
                article_id=data.article_id,
                pdf_services_client_id=data.pdf_services_client_id,
                pdf_services_client_secret=data.pdf_services_client_secret,
            )

            if loop is not None and loop.is_running():
                # 已有运行中的事件循环，用 run_until_complete 避免冲突
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    img_result = pool.submit(
                        lambda: asyncio.run(
                            handle_article_extract_images(extract_req)
                        )
                    ).result()
            else:
                img_result = asyncio.run(
                    handle_article_extract_images(extract_req)
                )
            # DOC-END id=handlers/bloggen/async-safe#1
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

            # --------------------------
            # 步骤2：清理无用图片、重编号剩余图片、更新博客引用
            # --------------------------
            # DOC-BEGIN id=handlers/bloggen/post-gen-image-cleanup#3 type=behavior v=3
            # summary: LLM生成初始博客后，清理extracted文件夹内未被引用的图片，对保留的图片重新生成连续编号，同步更新博客内的[[FIG:x]]引用
            # intent: 符合要求只保留用到的图片，永久固定编号，后续不再修改；避免存储无用图片占用空间，保证前端展示的图片编号连续
            used_figs = result["used_figs"]
            used_fig_nums = [int(f) for f in used_figs]
            logger.info(f"[BlogGen][{data.article_id}] Used original figure numbers: {used_fig_nums}")
            
            # 初始化图片存储文件夹
            article_stem = os.path.splitext(abs_path)[0]
            output_dir = article_stem + "_extracted"
            
            # DOC-BEGIN id=handlers/bloggen/clean-extracted-before-renumber#2 type=behavior v=2
            # summary: 在重编号图片前，彻底清理extracted文件夹，只保留structuredData.json，其他文件全部删除
            # intent: 确保extracted文件夹内严格只有图片（重编号后的）和structuredData.json，
            #   避免旧图片（包括table_x.png等）或其他文件残留。
            if os.path.exists(output_dir):
                for fname in os.listdir(output_dir):
                    if fname == "structuredData.json":
                        continue
                    fpath = os.path.join(output_dir, fname)
                    try:
                        if os.path.isfile(fpath):
                            os.remove(fpath)
                            logger.info(f"[BlogGen][{data.article_id}] Cleaned up file: {fpath}")
                    except Exception as e:
                        logger.warning(f"[BlogGen][{data.article_id}] Failed to cleanup {fpath}: {e}")
            else:
                os.makedirs(output_dir, exist_ok=True)
            # DOC-END id=handlers/bloggen/clean-extracted-before-renumber#2
            
            # 过滤出所有被博客引用的图片，保持原文章内的出现顺序
            filtered_images = [img for img in all_images if img["index"] in used_fig_nums]
            
            # 构建旧编号到新连续编号的映射
            old_to_new = {}
            for new_idx, img in enumerate(filtered_images, start=1):
                old_to_new[img["index"]] = new_idx
                
                # 更新图片自身的编号元数据
                img["index"] = new_idx
                img["filename"] = f"figure_{new_idx}.png"
                img["caption"] = f"Figure {new_idx}"
                
                # 保存重编号后的新图片文件
                try:
                    # 从base64中提取纯数据部分（去掉data:image/xxx;base64,前缀）
                    b64_data = img["base64_content"].split(",", 1)[1] if "," in img["base64_content"] else img["base64_content"]
                    img_bytes = base64.b64decode(b64_data)
                    new_file_path = os.path.join(output_dir, img["filename"])
                    with open(new_file_path, "wb") as f:
                        f.write(img_bytes)
                    logger.info(f"[BlogGen][{data.article_id}] Saved renumbered image: {new_file_path}")
                except Exception as e:
                    logger.warning(f"[BlogGen][{data.article_id}] Failed to save new image {img['filename']}: {e}")
            
            # 更新全局图片列表为过滤重编号后的版本
            all_images = filtered_images
            
            # 同步更新博客内的所有图片引用为新编号
            def _update_fig_refs(match):
                old_num = int(match.group(1))
                return f"[[FIG:{old_to_new.get(old_num, old_num)}]]"
            result["blog_markdown"] = re.sub(r'\[\[FIG:(\d+)\]\]', _update_fig_refs, result["blog_markdown"])
            
            # 更新used_figs为新编号
            new_used_figs = sorted([str(old_to_new[int(f)]) for f in used_figs if int(f) in old_to_new])
            result["used_figs"] = new_used_figs
            logger.info(f"[BlogGen][{data.article_id}] Renumbered figures, new used figures: {new_used_figs}")
            # DOC-END id=handlers/bloggen/post-gen-image-cleanup#3

            # --------------------------
            # 步骤3：生成结构化Figure Index（批量处理）
            # --------------------------
            # DOC-BEGIN id=handlers/bloggen/figure-index-batch-gen#2 type=behavior v=2
            # summary: 使用批量处理方式生成Figure Index：将所有用到的图片按FIGURE_BATCH_SIZE分批，
            #   每批独立调用generate_figure_index_batch函数生成注释，最后合并结果。
            #   批次内LLM输出临时编号1,2,...,N，后处理时映射回新编号（重编号后的连续编号）。
            # intent: 避免单次发送过多图片导致token超限；批次间完全独立，串行执行，
            #   任一批次失败直接终止整个Figure Index生成流程。
            #   重要：使用重编号后的新编号（1,2,3...），而不是原始编号。
            figure_index = None
            if new_used_figs:
                logger.info(f"[BlogGen][{data.article_id}] Starting batch Figure Index generation, used figures (renumbered): {new_used_figs}")

                # 收集博客中用到的图片，使用重编号后的新编号（1,2,3...）
                # 注意：new_used_figs 已经是重编号后的新编号列表 ['1','2','3','4','5','6','7','8']
                used_fig_nums_new = sorted([int(f) for f in new_used_figs])
                filtered_used_images = []
                for num in used_fig_nums_new:
                    for img in all_images:
                        if img["index"] == num:  # 这里img["index"]已经是重编号后的新编号
                            filtered_used_images.append(img)
                            break
                
                # 计算批次数量：平均分为3批，每批最少1张
                total_figs = len(used_fig_nums_new)
                batch_size = max(1, (total_figs + 2) // 3)  # 向上取整分3批
                total_batches = (total_figs + batch_size - 1) // batch_size
                logger.info(f"[BlogGen][{data.article_id}] Total {total_figs} figures, split into {total_batches} batches (batch_size={batch_size})")

                figure_index = []
                for batch_idx in range(total_batches):
                    # 计算当前批次的图片范围
                    start = batch_idx * FIGURE_BATCH_SIZE
                    end = min(start + FIGURE_BATCH_SIZE, len(used_fig_nums_new))
                    batch_new_nums = used_fig_nums_new[start:end]  # 使用重编号后的新编号
                    batch_images = filtered_used_images[start:end]

                    logger.info(f"[BlogGen][{data.article_id}] Processing batch {batch_idx + 1}/{total_batches}, figures (renumbered): {batch_new_nums}")

                    try:
                        batch_result = generate_figure_index_batch(
                            images_batch=batch_images,
                            article_title=title,
                            article_text=article_text_for_gen[:config.max_article_chars],
                            blog_markdown=result["blog_markdown"],
                            batch_original_nums=batch_new_nums,  # 传递重编号后的新编号
                            config=config,
                        )
                        figure_index.extend(batch_result)
                        logger.info(f"[BlogGen][{data.article_id}] Batch {batch_idx + 1} completed, generated {len(batch_result)} entries")
                    except Exception as e:
                        # 任一批次失败直接终止
                        logger.error(f"[BlogGen][{data.article_id}] Batch {batch_idx + 1} failed: {e}")
                        logger.error(f"[BlogGen][{data.article_id}] Terminating Figure Index generation due to batch failure")
                        figure_index = None
                        break

                # 保存Figure Index结果
                if figure_index is not None:
                    logger.info(f"[BlogGen][{data.article_id}] All batches completed, total {len(figure_index)} Figure Index entries")
            # DOC-END id=handlers/bloggen/figure-index-batch-gen#1

                # 保存到PDF对应的extracted文件夹
                pdf_stem = os.path.splitext(abs_path)[0]
                output_dir = pdf_stem + "_extracted"

                # DOC-BEGIN id=handlers/bloggen/save-figindex#1 type=behavior v=1
                # summary: 保存figindex.json到extracted文件夹，确保不覆盖已有的structuredData.json
                # intent: figindex.json是extracted文件夹中需要保留的两个json文件之一，
                #   另一个是structuredData.json。保存时确保目录存在，且不会清理其他文件。
                ensure_dir(output_dir)
                pdf_basename = os.path.splitext(os.path.basename(abs_path))[0]
                figindex_path = os.path.join(output_dir, f"{pdf_basename}.figindex.json")
                write_text_file(figindex_path, json.dumps(figure_index, ensure_ascii=False, indent=2))
                logger.info(f"[BlogGen][{data.article_id}] Saved structured Figure Index to {figindex_path}")
                # DOC-END id=handlers/bloggen/save-figindex#1
            else:
                logger.info(f"[BlogGen][{data.article_id}] No figures used, skipping Figure Index generation")

            # DOC-BEGIN id=handlers/bloggen/post-processing#1 type=behavior v=1
            # summary: 后处理阶段：如果提供了后处理LLM配置，则调用单模态LLM对已生成的blog进行精修和概念补充扩写。
            #   后处理LLM接收初始blog和原文（不包含图片），要求保持所有图片引用（FIG）不变。
            # intent: 初始blog由多模态LLM生成，可能遗漏某些概念或表达不够清晰；后处理LLM专注于文本精修，
            #   可以在不改变图片引用的前提下提升文章质量和概念解释深度。
            if data.refine_model_name and data.refine_api_key:
                logger.info(f"[BlogGen][{data.article_id}] Starting blog refinement with post-processing LLM...")
                refine_config = BlogGenConfig(
                    model_name=data.refine_model_name,
                    api_key=data.refine_api_key,
                    llm_url=data.refine_llm_url,
                    style=data.style or "math",
                )
                
                # 提取blog中实际引用的图片编号
                used_figs = set()
                for match in re.findall(r'\[\[FIG:(\d+)\]\]', result["blog_markdown"]):
                    used_figs.add(int(match))
                
                # 构建后处理LLM的提示词
                refine_prompt = f"""
你是一个专业的博客文章编辑。你的任务是对已生成的博客文章进行精修和概念补充扩写。

原始文章内容（供参考）：
{article_text_for_gen[:config.max_article_chars]}

初始博客文章（Markdown格式）：
{result["blog_markdown"]}

要求：
1. 保持博客文章的Markdown结构，包括标题、子标题、图片引用（如[[FIG:1]]）等。
2. 改进文章的语言表达，使其更加流畅、专业。
3. 对文章中的关键概念进行补充和扩写，提供更详细的解释和例子。
4. 确保所有图片引用（[[FIG:x]]）保持不变，不要添加或删除图片引用。
5. 如果原始文章中有重要的公式或数据，请确保在博客中得到准确的呈现。
6. 输出精修后的完整博客文章（Markdown格式），不要输出任何其他解释或标记。

注意：图片引用格式为[[FIG:x]]，x是数字，请确保这些引用在精修后的文章中保持不变 (注意格式必须是[[FIG:x]]而不是FIG:x)。
"""
                
                # 调用后处理LLM
                llm_refine = _new_llm(refine_config)
                refined_blog = llm_refine.query(refine_prompt, verbose=True)
                result["blog_markdown"] = refined_blog
                logger.info(f"[BlogGen][{data.article_id}] Blog refinement complete. Length: {len(refined_blog)} chars")
            # DOC-END id=handlers/bloggen/post-processing#1
            # DOC-END id=handlers/bloggen/extract-all#1

            # DOC-BEGIN id=handlers/bloggen/save-blog-and-images#1 type=behavior v=1
            # summary: 保存blog到txt文件，保存重编号后的images到images.json。
            #   此时extracted文件夹已经在post-gen-image-cleanup中清理完毕并保存了重编号图片，
            #   无需再次清理extracted文件夹。
            # intent: blog和images的保存与extracted文件夹的清理解耦，
            #   避免重复清理导致刚保存的图片被误删。
            blog_md = result["blog_markdown"]
            write_text_file(txt_path, blog_md)
            write_text_file(all_img_path, json.dumps(all_images, ensure_ascii=False, indent=2))
            logger.info(f"Complete generating blog for {title}. Saved blog and {len(all_images)} images.")
            # DOC-END id=handlers/bloggen/save-blog-and-images#1

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
