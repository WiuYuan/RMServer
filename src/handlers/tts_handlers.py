# src/handlers/tts_handlers.py
# ============================================================
# Handlers for TTS generation (parallel to article blog handlers)
# ============================================================

import os
import json
import base64
import asyncio
import traceback
import logging
import glob

from src.config import DATA_DIR, ARTICLES_ROOT
from src.utils.file_utils import read_text_file, write_text_file
from src.utils.article_helpers import (
    resolve_article_abs_path, load_raw_html, clean_html_for_injection,
)
from src.utils.article_tts_generator import generate_tts_from_blog, TTSGenConfig
from src.services.tts_job_queue import TTS_JOB_QUEUE
from src.models.requests import ArticleGenerateTTSReq

logger = logging.getLogger(__name__)


# ============================================================
# Path helpers (TTS-specific, mirrors blog path helpers)
# ============================================================

# DOC-BEGIN id=tts-handlers/path-helpers#1 type=design v=1
# summary: TTS文件路径命名规则——基于.html路径派生：
#   .tts.mp3(音频), .tts.txt(TTS中间文本), .tts.lock(运行中), .tts.queued(排队中), .tts.error(失败)
# intent: 与blog的.txt/.lock/.queued/.error命名平行但加.tts前缀避免冲突；
#   同一篇文章可以同时有blog和tts两套独立的状态文件
def _tts_audio_path(article_abs_path: str) -> str:
    base, _ = os.path.splitext(article_abs_path)
    return base + ".tts.mp3"


def _tts_text_path(article_abs_path: str) -> str:
    base, _ = os.path.splitext(article_abs_path)
    return base + ".tts.txt"


def _tts_lock_path(article_abs_path: str) -> str:
    base, _ = os.path.splitext(article_abs_path)
    return base + ".tts.lock"


def _tts_queued_path(article_abs_path: str) -> str:
    base, _ = os.path.splitext(article_abs_path)
    return base + ".tts.queued"


def _tts_error_path(article_abs_path: str) -> str:
    base, _ = os.path.splitext(article_abs_path)
    return base + ".tts.error"
# DOC-END id=tts-handlers/path-helpers#1


# ============================================================
# Cancel all TTS jobs
# ============================================================

# DOC-BEGIN id=tts-handlers/cancel-all#1 type=behavior v=1
# summary: 遍历ARTICLES_ROOT下所有.tts.lock/.tts.queued/.tts.error文件并删除，
#   清空TTS_JOB_QUEUE中尚未开始的任务；返回删除的文件列表和错误列表
# intent: 与handle_article_cancel_all_blogs完全对称的实现，清除所有TTS中间状态
async def handle_tts_cancel_all():
    root = os.path.abspath(ARTICLES_ROOT)
    removed = []
    errors = []

    for pattern in ("**/*.tts.lock", "**/*.tts.queued", "**/*.tts.error"):
        for fpath in glob.glob(os.path.join(root, pattern), recursive=True):
            try:
                os.remove(fpath)
                removed.append(fpath)
            except Exception as e:
                errors.append({"path": fpath, "error": str(e)})

    TTS_JOB_QUEUE.clear()

    return {
        "ok": True,
        "status": "cancelled",
        "removed_count": len(removed),
        "removed": removed,
        "errors": errors,
    }
# DOC-END id=tts-handlers/cancel-all#1


# ============================================================
# Delete TTS for a specific article
# ============================================================

# DOC-BEGIN id=tts-handlers/delete-tts#1 type=behavior v=1
# summary: 删除指定文章的所有TTS相关文件（.tts.mp3/.tts.txt/.tts.lock/.tts.queued/.tts.error），
#   返回删除的文件列表
# intent: 用户可能想重新生成TTS（换说话人/换风格），需要先删除旧文件重置状态
async def handle_tts_delete(article_id: str):
    abs_path = resolve_article_abs_path(article_id)

    audio_path = _tts_audio_path(abs_path)
    text_path = _tts_text_path(abs_path)
    lock_path = _tts_lock_path(abs_path)
    err_path = _tts_error_path(abs_path)
    queue_path = _tts_queued_path(abs_path)

    removed = []
    errors = []

    def _rm(p: str):
        try:
            if os.path.exists(p):
                os.remove(p)
                removed.append(p)
        except Exception as e:
            errors.append({"path": p, "error": str(e)})

    _rm(audio_path)
    _rm(text_path)
    _rm(lock_path)
    _rm(err_path)
    _rm(queue_path)

    return {
        "ok": True,
        "status": "deleted",
        "article_id": article_id,
        "removed": removed,
        "errors": errors,
    }
# DOC-END id=tts-handlers/delete-tts#1


# ============================================================
# Generate TTS (main handler, mirrors handle_article_generate_blog)
# ============================================================

# DOC-BEGIN id=tts-handlers/generate-tts#1 type=function v=1
# summary: TTS生成主handler，状态机与blog生成完全对称：
#   completed(返回音频base64+tts文本) → running → queued → failed → 新入队；
#   前置条件是blog必须已经生成（.txt存在），否则返回错误
# intent: 用户必须先生成blog再生成TTS，因为TTS是基于blog内容改写的；
#   音频以base64返回给前端，前端可以直接用<audio>标签播放；
#   tts_text一起返回方便前端展示"朗读稿"
async def handle_tts_generate(data: ArticleGenerateTTSReq):
    """
    状态机：
    - completed : TTS 已生成，返回音频和文本
    - running   : 正在生成（.tts.lock 存在）
    - queued    : 已入队等待（.tts.queued 存在）
    - failed    : 生成失败（.tts.error 存在）
    - queued    : 本次刚入队（ACK）
    """

    # --------------------------------------------------
    # 0. Resolve paths
    # --------------------------------------------------
    abs_path = resolve_article_abs_path(data.article_id)
    blog_txt_path = os.path.splitext(abs_path)[0] + ".txt"

    audio_path = _tts_audio_path(abs_path)
    text_path = _tts_text_path(abs_path)
    lock_path = _tts_lock_path(abs_path)
    err_path = _tts_error_path(abs_path)
    queue_path = _tts_queued_path(abs_path)

    # --------------------------------------------------
    # 0.5. Blog must exist first
    # --------------------------------------------------
    # DOC-BEGIN id=tts-handlers/require-blog#1 type=behavior v=1
    # summary: 检查blog的.txt文件是否存在，不存在则返回错误要求用户先生成blog
    # intent: TTS是基于blog内容生成的，没有blog就没有输入；
    #   这个检查在所有状态判断之前，确保即使TTS状态文件存在（如残留的.error），
    #   如果blog被删除了也会提示用户重新生成blog
    if not os.path.exists(blog_txt_path):
        return {
            "ok": False,
            "status": "error",
            "error": "Blog has not been generated yet. Please generate blog first.",
        }
    # DOC-END id=tts-handlers/require-blog#1

    # --------------------------------------------------
    # 1. COMPLETED
    # --------------------------------------------------
    if os.path.exists(audio_path):
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        tts_text = ""
        if os.path.exists(text_path):
            tts_text = read_text_file(text_path)

        return {
            "ok": True,
            "status": "completed",
            "audio_base64": audio_b64,
            "audio_format": "mp3",
            "tts_text": tts_text,
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
    # 5. Enqueue
    # --------------------------------------------------
    write_text_file(queue_path, json.dumps({
        "article_id": data.article_id,
    }, ensure_ascii=False))

    # DOC-BEGIN id=tts-handlers/worker#1 type=behavior v=1
    # summary: TTS worker函数：从queued→running→completed/failed，结构与blog worker对称；
    #   读取blog文本和文章原文 → 调用generate_tts_from_blog完整管线 → 写入.tts.mp3和.tts.txt
    # intent: worker在TTS_JOB_QUEUE的单worker线程中执行，与主线程隔离；
    #   logger输出会进入主线程的stderr，方便实时监控；
    #   finally块确保.lock清理和兜底.error写入，防止状态机卡死
    def worker():
        logger.info(f"[TTSWorker] Entered worker for article_id={data.article_id}")

        # queued → running
        try:
            os.remove(queue_path)
        except FileNotFoundError:
            pass

        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            logger.warning(f"[TTSWorker] .tts.lock already exists, aborting")
            return

        try:
            # Read blog text
            blog_markdown = read_text_file(blog_txt_path)
            logger.info(f"[TTSWorker] Blog text loaded, length={len(blog_markdown)} chars")

            # Read article text
            title, raw_html = load_raw_html(abs_path)
            article_text = clean_html_for_injection(raw_html)
            logger.info(f"[TTSWorker] Article text loaded: {title}")

            config = TTSGenConfig(
                model_name=data.model_name,
                api_key=data.api_key,
                llm_url=data.llm_url,
                fish_api_key=data.fish_api_key,
                reference_id=data.reference_id,
            )

            result = generate_tts_from_blog(
                article_id=data.article_id,
                article_text=article_text,
                blog_markdown=blog_markdown,
                config=config,
            )

            # Save TTS text
            write_text_file(text_path, result["tts_text"])
            logger.info(f"[TTSWorker] Saved TTS text to {text_path}")

            # Save audio
            with open(audio_path, "wb") as f:
                f.write(result["audio_bytes"])
            logger.info(f"[TTSWorker] Saved audio to {audio_path}, "
                        f"size={len(result['audio_bytes'])} bytes")

        except Exception as e:
            traceback.print_exc()
            logger.error(f"[TTSWorker] TTS generation failed for article_id={data.article_id}: {e}")
            write_text_file(err_path, str(e))

        finally:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass
            if not os.path.exists(audio_path) and not os.path.exists(err_path):
                try:
                    write_text_file(err_path, "Unknown error: TTS generation failed without details")
                except Exception:
                    pass
    # DOC-END id=tts-handlers/worker#1

    TTS_JOB_QUEUE.submit(worker)

    return {"ok": True, "status": "queued"}
# DOC-END id=tts-handlers/generate-tts#1


# ============================================================
# Get TTS status for article tree (used by article_list)
# ============================================================

# DOC-BEGIN id=tts-handlers/get-tts-status#1 type=function v=1
# summary: 给定文章的html绝对路径，检查.tts.mp3/.tts.lock/.tts.queued/.tts.error文件，
#   返回TTS状态字符串：completed/running/queued/failed/none
# intent: 供build_articles_tree调用，在文章列表中附带TTS状态，
#   让前端知道哪些文章已有音频、哪些正在生成
def get_tts_status(article_abs_path: str) -> str:
    base, _ = os.path.splitext(article_abs_path)
    if os.path.exists(base + ".tts.mp3"):
        return "completed"
    if os.path.exists(base + ".tts.lock"):
        return "running"
    if os.path.exists(base + ".tts.queued"):
        return "queued"
    if os.path.exists(base + ".tts.error"):
        return "failed"
    return "none"
# DOC-END id=tts-handlers/get-tts-status#1