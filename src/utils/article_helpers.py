# src/utils/article_helpers.py
import os
import re
from typing import Dict, Any, Optional, Tuple
from fastapi import HTTPException
import trafilatura
from bs4 import BeautifulSoup
from src.utils.file_utils import ensure_dir
from src.config import ARTICLES_ROOT

def resolve_article_abs_path(article_id: str) -> str:
    """
    article_id: 相对 ARTICLES_ROOT 的路径，如 "physics/ai_in_phy/a.html" 或 "physics/ai_in_phy/a.pdf"
    返回: 绝对路径
    """
    rel = (article_id or "").lstrip("/").strip()
    abs_path = os.path.abspath(os.path.join(ARTICLES_ROOT, rel))
    root = os.path.abspath(ARTICLES_ROOT) + os.sep

    # 防止路径穿越
    if not abs_path.startswith(root):
        raise HTTPException(status_code=400, detail="Invalid article_id path traversal")

    # 必须存在且是文件
    if not (os.path.exists(abs_path) and os.path.isfile(abs_path)):
        raise HTTPException(status_code=404, detail=f"Article file not found: {article_id}")

    # 允许 .html 和 .pdf 文件
    allowed_extensions = {".html", ".pdf"}
    file_ext = os.path.splitext(abs_path.lower())[1]
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Only {', '.join(allowed_extensions)} files are supported")

    return abs_path

def build_articles_tree(root_dir: str = ARTICLES_ROOT) -> Dict[str, Any]:
    root_dir = os.path.abspath(root_dir)

    def walk(cur_abs: str, cur_rel: str) -> Dict[str, Any]:
        name = os.path.basename(cur_abs) if cur_rel else os.path.basename(root_dir)
        node = {"type": "dir", "name": name, "path": cur_rel, "children": []}

        try:
            entries = sorted(os.listdir(cur_abs))
        except Exception:
            return node

        for ent in entries:
            if ent.startswith("."):
                continue
            abs_p = os.path.join(cur_abs, ent)
            rel_p = os.path.join(cur_rel, ent) if cur_rel else ent

            if os.path.isdir(abs_p):
                node["children"].append(walk(abs_p, rel_p))
            else:
                file_ext = ent.lower()
                # 支持 .html 和 .pdf 文件
                if file_ext.endswith(".html") or file_ext.endswith(".pdf"):
                    # DOC-BEGIN id=helpers/articles/tree-blog-status#1 type=behavior v=1
                    # summary: 根据 .html 或 .pdf 同名的 .txt/.lock/.queued/.error 文件是否存在，
                    #   计算该文章的 blog_status 并附加到 file 节点上；
                    #   优先级：txt(completed) > lock(running) > queued(queued) > error(failed) > none
                    # intent: 前端 fetchArticles 遍历 tree 初始化 blogStatuses Map，
                    #   如果后端不在 tree 中返回 blog_status，刷新页面后所有按钮都显示蓝色（未生成），
                    #   与实际状态不符；检查顺序按状态优先级排列，completed 最优先，
                    #   因为 .lock 可能因进程崩溃残留但 .txt 已写入
                    base_no_ext = os.path.splitext(abs_p)[0]
                    if os.path.exists(base_no_ext + ".txt"):
                        blog_status = "completed"
                    elif os.path.exists(base_no_ext + ".lock"):
                        blog_status = "running"
                    elif os.path.exists(base_no_ext + ".queued"):
                        blog_status = "queued"
                    elif os.path.exists(base_no_ext + ".error"):
                        blog_status = "failed"
                    else:
                        blog_status = "none"
                    # DOC-END id=helpers/articles/tree-blog-status#1

                    # DOC-BEGIN id=helpers/articles/tree-tts-status#1 type=behavior v=1
                    # summary: 检查TTS状态文件(.tts.mp3/.tts.lock/.tts.queued/.tts.error)，
                    #   计算tts_status并附加到file节点，逻辑与blog_status完全对称
                    # intent: 前端需要在文章列表中展示TTS生成状态，与blog状态独立显示
                    if os.path.exists(base_no_ext + ".tts.mp3"):
                        tts_status = "completed"
                    elif os.path.exists(base_no_ext + ".tts.lock"):
                        tts_status = "running"
                    elif os.path.exists(base_no_ext + ".tts.queued"):
                        tts_status = "queued"
                    elif os.path.exists(base_no_ext + ".tts.error"):
                        tts_status = "failed"
                    else:
                        tts_status = "none"
                    # DOC-END id=helpers/articles/tree-tts-status#1

                    # 获取文件扩展名和标题
                    if file_ext.endswith(".html"):
                        title = ent[:-5]  # 移除 .html
                        file_type = "html"
                    else:  # .pdf
                        title = ent[:-4]  # 移除 .pdf
                        file_type = "pdf"

                    node["children"].append({
                        "type": "file",
                        "name": ent,
                        "article_id": rel_p.replace("\\", "/"),
                        "title": title,
                        "file_type": file_type,
                        "blog_status": blog_status,
                        "tts_status": tts_status,
                    })
        return node

    ensure_dir(root_dir)
    return walk(root_dir, "")

def sanitize_filename(s: str) -> str:
    """仅用于没有 ID 时生成默认文件名"""
    s = re.sub(r"[^\w\-\u4e00-\u9fff\s\.]+", "", (s or "").strip())
    return s[:100] or "article"

def clean_html_for_injection(raw_html: str) -> str:
    text = trafilatura.extract(raw_html, include_tables=True, include_comments=False)
    return text or ""

def load_raw_html(abs_path: str) -> Tuple[str, str]:
    filename = os.path.basename(abs_path)
    title = filename[:-5] if filename.lower().endswith(".html") else filename
    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    return title, html

def clean_html_for_view(raw_html: str, title: Optional[str] = None) -> str:
    """
    返回：可直接 iframe 展示的「阅读级 HTML」
    """

    # 1. 用 trafilatura 抽正文（返回的是 HTML 片段）
    extracted = trafilatura.extract(
        raw_html,
        include_tables=True,
        include_comments=False,
        output_format="html"
    )

    if not extracted:
        extracted = "<p>(No readable content)</p>"

    # 2. 用 BeautifulSoup 再清一遍
    soup = BeautifulSoup(extracted, "lxml")

    # 移除潜在危险 / 无用标签（保险）
    for tag in soup.find_all(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    body_html = soup.prettify()
    title_html = f"<h1 class='article-title'>{title}</h1>" if title else ""

    # 3. 包一层你可控的 HTML + CSS
    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>

<style>
/* ===== Reading Style ===== */
body {{
  margin: 0;
  padding: 16px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
               Helvetica, Arial, sans-serif;
  line-height: 1.65;
  background: #ffffff;
  color: #111;
}}

.article {{
  max-width: 900px;
  margin: 0 auto;
}}

h1, h2, h3 {{
  line-height: 1.3;
}}

img {{
  max-width: 100%;
  height: auto;
}}

table {{
  border-collapse: collapse;
  width: 100%;
}}

th, td {{
  border: 1px solid #ccc;
  padding: 6px 8px;
}}
</style>
</head>

<body>
<div class="article">
{title_html}
{body_html}
</div>
</body>
</html>
"""


def article_txt_path_from_html(article_abs_path: str) -> str:
    """
    /path/to/abc.html -> /path/to/abc.txt
    """
    base, _ = os.path.splitext(article_abs_path)
    return base + ".txt"

def article_lock_path(article_abs_path: str) -> str:
    base, _ = os.path.splitext(article_abs_path)
    return base + ".lock"

def article_error_path(article_abs_path: str) -> str:
    base, _ = os.path.splitext(article_abs_path)
    return base + ".error"

def article_queued_path(article_abs_path: str) -> str:
    base, _ = os.path.splitext(article_abs_path)
    return base + ".queued"
