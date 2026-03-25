
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

logger = logging.getLogger(__name__)


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

async def handle_article_extract_images(data: ArticleExtractImagesReq):
    abs_path = resolve_article_abs_path(data.article_id)

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
            title, raw_html = load_raw_html(abs_path)
            logger.info(f"[Worker] Start generating blog for: {title}")
            cleaned_text = clean_html_for_injection(raw_html)

            img_result = asyncio.run(
                handle_article_extract_images(
                    ArticleExtractImagesReq(article_id=data.article_id)
                )
            )
            all_images = img_result.get("images", [])
            write_text_file(all_img_path, json.dumps(all_images, ensure_ascii=False, indent=2))

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

            result = generate_blog_from_article_tree(
                task_id=data.task_id,
                article_id=data.article_id,
                article_title=title,
                article_text=cleaned_text,
                image_catalog=image_catalog,
                config=config,
                tc=tc,
            )

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
