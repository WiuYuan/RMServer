import os
import json
import asyncio
import threading
import time
import traceback
import hashlib
import logging
from typing import Dict, Any

from src.config import DATA_DIR, ARTICLES_ROOT
from src.utils.file_utils import (
    ensure_dir, read_text_file, write_text_file,
)
from src.utils.article_helpers import (
    resolve_article_abs_path, build_articles_tree,
    clean_html_for_injection, clean_html_for_view, load_raw_html,
    article_txt_path_from_html, article_lock_path, article_error_path,
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

logger = logging.getLogger(__name__)

# DOC-BEGIN id=manual/bloggen/global-serial#1 type=design v=1
# summary: 提供全局串行化锁，保证同一时刻仅有一个文章博客生成任务在运行
# intent: 通过进程内互斥避免并发 submit 多个生成任务导致队列/日志等出现“max entries”类上限问题；该锁只约束“生成”流程，不影响已完成结果的读取
BLOG_GEN_GLOBAL_LOCK = threading.Lock()
BLOG_GEN_GLOBAL_STATE: Dict[str, Any] = {}
# DOC-END id=manual/bloggen/global-serial#1


async def handle_article_get_html(data: ArticleGetHtmlReq):
    abs_path = resolve_article_abs_path(data.article_id)
    title, html = load_raw_html(abs_path)
    return {"ok": True, "article_id": data.article_id, "title": title, "clean_html": clean_html_for_view(html, title)}
    # ↑字段名你前端叫 clean_html，这里为了不改前端，继续用 clean_html 装“原始 html”

async def inject_article_to_task(task_id: str, article_id: str, title: str, content: str):
    """把文章内容注入到 LLM 上下文"""
    task_dir = f"{DATA_DIR}/tasks/{task_id}"
    ensure_dir(task_dir)
    tc = Tool_Calls(LOG_DIR=task_dir, MAX_CHAR=800000, mode="Summary")

    # 构造注入消息
    new_tool_calls = [
        {
            "role": "assistant", # 使用 assistant 角色模拟
            "content": (
                f"[System Context Injection]\n"
                f"User has selected an external article.\n"
                f"Title: {title}\n"
                f"Filename: {article_id}\n\n"
                f"Content Start:\n"
                f"{content[:60000]}\n" # 截断防止爆 Token
                f"Content End.\n"
            )
        }
    ]
    tc.extend(new_tool_calls)

# === 业务 Handlers ===
async def handle_article_delete_blog(data: ArticleDeleteBlogReq):
    abs_path = resolve_article_abs_path(data.article_id)

    txt_path = article_txt_path_from_html(abs_path)
    lock_path = article_lock_path(abs_path)
    err_path = article_error_path(abs_path)

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

    # 删除生成结果与缓存
    _rm(txt_path)
    _rm(all_img_path)
    _rm(err_path)

    # 无条件解锁
    _rm(lock_path)

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

    # ✅ 从“完整 HTML 源码”建立 CSS 变量映射（SingleFile 的图一般在这里）
    css_var_map = _build_css_var_map(raw_html)
    # print("css vars sample:", list(css_var_map.items())[:1])

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

        # 去重
        src_hash = hashlib.md5(clean_src.encode()).hexdigest()
        if src_hash in seen_src_hashes:
            continue

        # 尺寸
        w, h = _get_size_from_data_uri(clean_src)
        
        # print("IMG", count, "size=", w, h, "src=", clean_src[:40])

        # ✅ 2. 如果解码失败，再 fallback 到 HTML 属性
        if w is None or h is None:
            w2, h2 = _get_img_size(img)
            w = w if w is not None else w2
            h = h if h is not None else h2

        # ✅ 3. 只有在“明确知道尺寸”时才过滤
        print(f"[IMAGE] width={w}, height={h}, total={w+h}")
        if w is not None and h is not None:
            if w < MIN_WIDTH or h < MIN_HEIGHT or w + h < MIN_TOTAL:
                continue

        # 类型
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
    # 你说 list 就返回目录结构，所以 status 参数可以忽略或保留兼容
    return {"ok": True, "tree": build_articles_tree(ARTICLES_ROOT)}

async def handle_task_select_article(data: TaskSelectArticleReq):
    abs_path = resolve_article_abs_path(data.article_id)
    title, raw_html = load_raw_html(abs_path)

    cleaned = clean_html_for_injection(raw_html)

    # 注入：用 cleaned（不是 raw）
    await inject_article_to_task(
        task_id=data.task_id,
        article_id=data.article_id,  # 这里用相对路径当“文件名”
        title=title,
        content=cleaned,
    )
    return {"ok": True, "msg": f"Article '{title}' injected successfully."}

async def handle_article_generate_blog(data: ArticleGenerateBlogReq):
    """
    Unified command-style blog generator.

    Status machine:
    - completed : return blog + images immediately
    - running   : background worker is running
    - failed    : generation failed
    - started   : background worker just started
    """

    # --------------------------------------------------
    # 0. Resolve paths
    # --------------------------------------------------
    abs_path = resolve_article_abs_path(data.article_id)

    txt_path = article_txt_path_from_html(abs_path)
    lock_path = article_lock_path(abs_path)
    err_path = article_error_path(abs_path)

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

        # persist ALL images
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
        return {
            "ok": True,
            "status": "running",
        }

    # --------------------------------------------------
    # 3. FAILED
    # --------------------------------------------------
    if os.path.exists(err_path):
        return {
            "ok": False,
            "status": "failed",
            "error": read_text_file(err_path),
        }

    # --------------------------------------------------
    # 4. Acquire lock
    # --------------------------------------------------
    # DOC-BEGIN id=manual/bloggen/enforce-serial#1 type=behavior v=2
    # summary: 强制博客生成任务全局串行化：上一个未完成时拒绝开启下一个（跨进程 + 进程内）
    # intent: 之前仅用 threading.Lock 只能约束“单进程”；当 uvicorn/gunicorn 开多 worker 时会并发触发，仍可能导致队列/日志出现 max entries 等上限问题。
    # intent: 这里使用“全局锁文件（O_EXCL 原子创建）”实现跨进程互斥，并辅以进程内互斥避免同进程并发；返回 busy 让前端轮询等待，而不是继续 submit。
    global_lock_path = os.path.join(DATA_DIR, ".bloggen_global.lock")
    global_locked = False

    lock_payload = {
        "task_id": data.task_id,
        "article_id": data.article_id,
        "started_at": time.time(),
        "pid": os.getpid(),
    }

    # try:
    #     fd_global = os.open(global_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    #     os.write(fd_global, json.dumps(lock_payload, ensure_ascii=False).encode("utf-8"))
    #     os.close(fd_global)
    #     global_locked = True
    # except FileExistsError:
    #     current_job: Dict[str, Any] = {}
    #     try:
    #         raw = read_text_file(global_lock_path).strip()
    #         current_job = json.loads(raw) if raw else {}
    #     except Exception:
    #         current_job = {}
    #     return {
    #         "ok": True,
    #         "status": "busy",
    #         "current_job": current_job,
    #     }

    if not BLOG_GEN_GLOBAL_LOCK.acquire(blocking=False):
        try:
            os.remove(global_lock_path)
        except FileNotFoundError:
            pass
        return {
            "ok": True,
            "status": "busy",
            "current_job": lock_payload,
        }

    BLOG_GEN_GLOBAL_STATE.clear()
    BLOG_GEN_GLOBAL_STATE.update(lock_payload)
    # DOC-END id=manual/bloggen/enforce-serial#1

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        BLOG_GEN_GLOBAL_STATE.clear()
        try:
            BLOG_GEN_GLOBAL_LOCK.release()
        except RuntimeError:
            pass
        if global_locked:
            try:
                os.remove(global_lock_path)
            except FileNotFoundError:
                pass
        return {
            "ok": True,
            "status": "running",
        }

    # --------------------------------------------------
    # 5. Background worker
    # --------------------------------------------------
    def worker():
        try:
            # ---------- Load article ----------
            title, raw_html = load_raw_html(abs_path)
            logger.info(f"Start generating blog for {title}.")
            cleaned_text = clean_html_for_injection(raw_html)

            # ---------- Extract images (ONLY HERE) ----------
            img_result = asyncio.run(
                handle_article_extract_images(
                    ArticleExtractImagesReq(article_id=data.article_id)
                )
            )
            all_images = img_result.get("images", [])

            # persist ALL images
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

            # ---------- Blog config ----------
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

            # ---------- Heavy generation ----------
            future = BLOG_JOB_QUEUE.submit(
                lambda: generate_blog_from_article_tree(
                    task_id=data.task_id,
                    article_id=data.article_id,
                    article_title=title,
                    article_text=cleaned_text,
                    image_catalog=image_catalog,
                    config=config,
                    tc=tc,
                )
            )
            
            result = future.result()

            blog_md = result["blog_markdown"]

            # ---------- Persist ----------
            write_text_file(txt_path, blog_md)
            logger.info(f"Complete generating blog for {title}.")

        except Exception as e:
            traceback.print_exc()
            write_text_file(err_path, str(e))

        finally:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass

            # DOC-BEGIN id=manual/bloggen/release-serial#1 type=behavior v=2
            # summary: 生成任务结束时释放全局串行化锁并清理运行态（同时释放跨进程锁文件）
            # intent: 必须在成功/失败两条路径都释放锁，否则会造成“永久 busy”；同时删除全局锁文件以允许其他 worker/进程继续提交；对重复释放做兜底避免异常路径二次 release。
            BLOG_GEN_GLOBAL_STATE.clear()
            try:
                BLOG_GEN_GLOBAL_LOCK.release()
            except RuntimeError:
                pass

            try:
                os.remove(global_lock_path)
            except FileNotFoundError:
                pass
            # DOC-END id=manual/bloggen/release-serial#1

    threading.Thread(target=worker, daemon=True).start()

    # --------------------------------------------------
    # 6. Immediate ACK
    # --------------------------------------------------
    return {
        "ok": True,
        "status": "started",
    }
