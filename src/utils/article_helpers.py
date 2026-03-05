import os
import re
from typing import Dict, Any, Optional, Tuple
from fastapi import HTTPException
import trafilatura
from bs4 import BeautifulSoup
from src.utils.file_utils import ensure_dir

ARTICLES_ROOT = "/home/ubuntu/workspace/data/articles"

def resolve_article_abs_path(article_id: str) -> str:
    """
    article_id: 相对 ARTICLES_ROOT 的路径，如 "physics/ai_in_phy/a.html"
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

    # 可选：只允许 .html
    if not abs_path.lower().endswith(".html"):
        raise HTTPException(status_code=400, detail="Only .html is supported")

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
                if ent.lower().endswith(".html"):
                    node["children"].append({
                        "type": "file",
                        "name": ent,
                        "article_id": rel_p.replace("\\", "/"),  # 统一成 url 风格
                        "title": ent[:-5],
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
