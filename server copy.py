# server.py
import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.services.llm import LLM 
import os
import asyncio
import queue
from src.services.external_client import ExternalClient
from fastapi.responses import StreamingResponse
from src.services.agents import Tool_Calls, stop_if_no_tool_calls
import traceback
from functools import partial, update_wrapper
from src.services.custom_tools import custom_tools
import threading, time
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any, Optional, Union, Literal
from pydantic import TypeAdapter
import shutil, json, re
from bs4 import BeautifulSoup
import trafilatura
from src.services.session_history import history_manager
import hashlib
from src.utils.article_blog_generator import (
    generate_blog_from_article_tree,
    BlogGenConfig,
)
from src.services.blog_job_queue import BLOG_JOB_QUEUE
import base64
from io import BytesIO
from PIL import Image
import logging

from src.utils.playbook_store import (
    PlaybookStore,
    PlaybookListReq,
    PlaybookGetReq,
    PlaybookUpsertPolicyReq,
    PlaybookUpsertBacklogReq,
    PlaybookUpsertAssumptionReq,
    PlaybookDeleteReq,
    handle_playbook_list,
    handle_playbook_get,
    handle_playbook_upsert_policy,
    handle_playbook_upsert_backlog,
    handle_playbook_upsert_assumption,
    handle_playbook_delete,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# DOC-BEGIN id=manual/bloggen/global-serial#1 type=design v=1
# summary: 提供全局串行化锁，保证同一时刻仅有一个文章博客生成任务在运行
# intent: 通过进程内互斥避免并发 submit 多个生成任务导致队列/日志等出现“max entries”类上限问题；该锁只约束“生成”流程，不影响已完成结果的读取
BLOG_GEN_GLOBAL_LOCK = threading.Lock()
BLOG_GEN_GLOBAL_STATE: Dict[str, Any] = {}
# DOC-END id=manual/bloggen/global-serial#1


# === 基础配置 ===
SECRET_FILE = "/home/ubuntu/.env_secret"
DATA_DIR = "/home/ubuntu/workspace/data"
ARTICLES_ROOT = f"{DATA_DIR}/articles"
PLAYBOOK_ROOT = f"{DATA_DIR}/playbook"
PLAYBOOK_ACTIONS_NEED_ROOT = {
    "playbook_list",
    "playbook_get",
    "playbook_upsert_policy",
    "playbook_upsert_backlog",
    "playbook_upsert_assumption",
    "playbook_delete",
    "playbook_suggest_next_step",
}

# DOC-BEGIN id=manual/relay/plan-storage-paths#1 type=design v=1
# summary: 定义“远端中转器”计划数据的落盘路径（DATA_DIR 下）
# intent: 你需要远端作为中转器供另一个前端拉取，因此必须落盘到稳定文件而非内存；
#   放在 DATA_DIR/relay 下便于与 articles/playbook/tasks 等数据分区隔离；
#   文件路径集中定义，便于未来扩展（例如加 settings、加多用户命名空间）。
RELAY_DIR = os.path.join(DATA_DIR, "relay")
RELAY_CURRENT_PLAN_FILE = os.path.join(RELAY_DIR, "current_plan.json")
# DOC-END id=manual/relay/plan-storage-paths#1

def with_default_playbook_root(action: str, data: dict) -> dict:
    """
    Keep request schemas unchanged, but allow clients to omit playbook_root.
    If missing/empty, inject server default PLAYBOOK_ROOT.
    """
    if not isinstance(data, dict):
        return data
    if action in PLAYBOOK_ACTIONS_NEED_ROOT:
        if not data.get("playbook_root"):
            data = {**data, "playbook_root": PLAYBOOK_ROOT}
    return data

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

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def sanitize_filename(s: str) -> str:
    """仅用于没有 ID 时生成默认文件名"""
    s = re.sub(r"[^\w\-\u4e00-\u9fff\s\.]+", "", (s or "").strip())
    return s[:100] or "article"

def clean_html_for_injection(raw_html: str) -> str:
    text = trafilatura.extract(raw_html, include_tables=True, include_comments=False)
    return text or ""


# === 数据模型 ===
class PlaybookSuggestNextStepReq(BaseModel):
    playbook_root: str = "/home/ubuntu/workspace/data/playbook"
    model_name: str
    api_key: str
    llm_url: Optional[str] = "https://api.deepseek.com/v1/chat/completions"

    assumptions: list[str] = []
    current_condition: str

# DOC-BEGIN id=manual/relay/remote-plan-models#1 type=design v=1
# summary: 远端中转器的“计划数据 set/get”请求模型
# intent: 该中转器只负责存取，不对 plan 的具体结构做强约束（避免前端迭代被后端 schema 卡死）；
#   仅增加可选 source 方便排查“是谁写入的”（例如主前端/测试脚本/另一个前端）。
class RemotePlanSetReq(BaseModel):
    plan: Any
    source: Optional[str] = None

class RemotePlanGetReq(BaseModel):
    pass
# DOC-END id=manual/relay/remote-plan-models#1

class ArticleDeleteBlogReq(BaseModel):
    article_id: str
    
class TaskGetTerminalStatusReq(BaseModel):
    task_id: str
    
class ArticleExtractImagesReq(BaseModel):
    article_id: str

class ArticleUpsertHTML(BaseModel):
    task_id: Optional[str] = None
    title: str
    html: str
    article_id: Optional[str] = None # 如果传了文件名（如 "test.html"），就用这个

class ArticleListReq(BaseModel):
    status: Optional[Literal["processing","completed","all"]] = "all"

class ArticleGetHtmlReq(BaseModel):
    article_id: str

class TaskSelectArticleReq(BaseModel):
    task_id: str
    article_id: str
    
class PinnedCode(BaseModel):
    hash: str
    filename: str
    content: str

class LLMRequestData(BaseModel):
    model_name: str
    question: str
    api_key: str
    llm_url: Optional[str] = "https://api.deepseek.com/v1/chat/completions"
    task_id: Optional[str] = "1"
    system_prompt_mode: Optional[str] = "default"
    enable_fc: Optional[bool] = False
    pinned_codes: Optional[list[PinnedCode]] = []

class ActionRequest(BaseModel):
    action: str
    data: dict

class StopData(BaseModel):
    task_id: str
    
class ArticleGenerateBlogReq(BaseModel):
    task_id: str
    article_id: str
    model_name: str
    api_key: str
    llm_url: Optional[str] = None
    style: Optional[Literal["math", "normal", "rigorous"]] = "math"


# === 核心辅助函数 ===
def extract_json_by_braces(text: str) -> dict:
    if not isinstance(text, str):
        text = str(text)

    start = text.find("{")
    if start < 0:
        raise ValueError("No '{' found in LLM output")

    end = text.rfind("}")
    if end < 0 or end <= start:
        raise ValueError("No valid '}' found in LLM output")

    json_text = text[start:end + 1]
    return json.loads(json_text)

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

def list_files_in_dir(root: str):
    out = []
    if not os.path.isdir(root): 
        return out
    
    try:
        # 按修改时间倒序（最新的在前）
        files = sorted(os.listdir(root), key=lambda x: os.path.getmtime(os.path.join(root, x)), reverse=True)
    except:
        files = os.listdir(root)

    for fname in files:
        # 只列出 html 文件
        if fname.lower().endswith(".html"):
            out.append({
                "article_id": fname,          # ID 就是文件名 (e.g. "random_name.html")
                "title": fname[:-5]           # Title 只是用来展示 (e.g. "random_name")
            })
    return out

def _parse_px(v: str | None) -> int | None:
    """
    把 '300', '300px', '30%' 等解析成像素（百分比直接忽略）
    """
    if not v:
        return None
    v = v.strip().lower()
    if v.endswith("px"):
        v = v[:-2]
    if v.isdigit():
        return int(v)
    return None


def _get_img_size(img):
    """
    尝试从 img 的属性或 style 中解析尺寸
    """
    # 1️⃣ HTML 属性
    w = _parse_px(img.get("width"))
    h = _parse_px(img.get("height"))

    # 2️⃣ style="width:xxx;height:xxx"
    if (w is None or h is None) and img.get("style"):
        style = img["style"]
        mw = re.search(r"width\s*:\s*(\d+)px", style)
        mh = re.search(r"height\s*:\s*(\d+)px", style)
        if w is None and mw:
            w = int(mw.group(1))
        if h is None and mh:
            h = int(mh.group(1))

    return w, h

def _get_size_from_data_uri(data_uri: str) -> tuple[int | None, int | None]:
    """
    从 data:image/...;base64,... 解码真实图片尺寸
    """
    try:
        if not data_uri.startswith("data:image"):
            return None, None

        # data:image/webp;base64,AAAA...
        header, b64 = data_uri.split(",", 1)
        raw = base64.b64decode(b64)

        with Image.open(BytesIO(raw)) as im:
            return im.width, im.height
    except Exception:
        return None, None
    
def _build_css_var_map(raw_html: str) -> dict[str, str]:
    """
    扫描整个 HTML 源码（包含 <style> 内联 CSS），提取：
    --sf-img-16: url(data:image/xxx;base64,....)
    返回 dict: {"--sf-img-16": "data:image/webp;base64,....", ...}
    """
    var_map: dict[str, str] = {}

    # 兼容 url("data:...") / url('data:...') / url(data:...)
    # 兼容有空格/换行/!important
    pattern = re.compile(
        r"(?P<name>--sf-img-[A-Za-z0-9_-]+)\s*:\s*url\(\s*"
        r"(?P<quote>['\"]?)"
        r"(?P<data>data:image[^'\"\)]+)"
        r"(?P=quote)\s*\)",
        re.IGNORECASE | re.DOTALL,
    )

    for m in pattern.finditer(raw_html):
        name = m.group("name")
        data = m.group("data").strip()
        # 有些 SingleFile 会把 data uriing 很长，中间夹换行；DOTALL 已覆盖，但这里再去掉空白更稳
        data = re.sub(r"\s+", "", data)
        var_map[name] = data

    return var_map

def _extract_real_image_data(img, css_var_map: dict[str, str]) -> str | None:
    """
    从 img 本体提取真正 data:image...；支持：
    1) src=data:image...(非svg)
    2) style 里 background-image:url(data:...)
    3) style 里 background-image:var(--sf-img-XX) -> 用 css_var_map 解析
    """
    # 1) src 直接是 data:image（过滤掉占位 svg）
    src = (img.get("src") or "").strip()
    if src.startswith("data:image") and not src.startswith("data:image/svg+xml"):
        return src

    style = img.get("style") or ""

    # 2) inline style 里有 url(data:...)
    m = re.search(r"background-image\s*:\s*url\(\s*(['\"]?)(data:image[^'\"\)]+)\1\s*\)",
                  style, flags=re.IGNORECASE | re.DOTALL)
    if m:
        data = re.sub(r"\s+", "", m.group(2).strip())
        return data

    # 3) inline style 里是 var(--sf-img-XX)
    m = re.search(r"background-image\s*:\s*var\(\s*(--sf-img-[A-Za-z0-9_-]+)\s*\)",
                  style, flags=re.IGNORECASE)
    if m:
        name = m.group(1)
        data = css_var_map.get(name)
        if data:
            return data

    return None


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

async def handle_playbook_suggest_next_step(data: PlaybookSuggestNextStepReq) -> dict:
    store = PlaybookStore(data.playbook_root)

    # 收集历史 policies（包含 when/next_step 供相似度判断）
    metas = store.list_policies()
    policies = []
    id_to_rel = {}

    for m in metas:
        id_to_rel.setdefault(m.id, []).append(m.rel_path)  # 兼容同名 id（不同文件夹）

        # ✅ 用 rel_path 读取（支持子文件夹）
        p = store.get_policy_by_rel(m.rel_path)

        policies.append({
            "id": m.id,
            "rel_path": m.rel_path,   # ✅ 关键：给 LLM 一个稳定可定位的引用
            "title": p.get("title", ""),
            "when": p.get("when", ""),
            "next_step": p.get("next_step", ""),
        })

    assumptions_text = "\n".join([f"- {a}" for a in (data.assumptions or [])]) or "- (none)"

    prompt = f"""
你是一个决策助手。请基于“假设”和“当前条件”，在给定历史 policy 列表中找到“when 描述最相似”的一条.

要求：
只输出 policy_id, 一个string（输出最相似 policy 的 rel_path；例如 "policies/xxx/P_123.json"；若都不匹配，输出空字符串）
不要输出任何其他东西, 不要输出双引号, 只输出一个rel_path

【假设】
{assumptions_text}

【当前条件】
{data.current_condition}

【历史 policies】（每条都含 id 与 rel_path，优先用 rel_path 做返回值）
{json.dumps(policies, ensure_ascii=False, indent=2)}
""".strip()

    llm = LLM(
        api_key=data.api_key,
        llm_url=data.llm_url,
        model_name=data.model_name,
        format="openai",
        ec=None,
    )

    policy_ref = llm.query(prompt)
    # obj = extract_json_by_braces(raw)

    # ✅ 如果 LLM 直接返回 rel_path（包含 / 或以 .json 结尾），直接用
    if policy_ref and ("/" in policy_ref or policy_ref.lower().endswith(".json")):
        chosen_rel_path = policy_ref
    else:
        # ✅ 否则当作 id，尝试映射到 rel_path（若重复，取第一个）
        cands = id_to_rel.get(policy_ref, [])
        chosen_rel_path = cands[0] if cands else ""

    # ✅ 额外返回：该 policy 文件中存储的原始 next_step（用于对照/溯源）
    policy_next_step = ""
    if chosen_rel_path:
        try:
            p = store.get_policy_by_rel(chosen_rel_path)
            policy_next_step = str(p.get("next_step") or "")
        except Exception:
            policy_next_step = ""

    return {
        "ok": True,
        "policy_id": chosen_rel_path,                 # rel_path
        "policy": policy_next_step,         # policy 原文 next_step
    }

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

def try_acquire_lock(lock_path: str) -> bool:
    """
    原子创建 lock 文件
    成功：返回 True
    已存在：返回 False
    """
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False
    
def release_lock(lock_path: str):
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass

def read_text_file(p: str) -> str:
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def write_text_file(p: str, content: str):
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)

# DOC-BEGIN id=manual/relay/json-atomic-write#1 type=logic v=1
# summary: 将 JSON 数据以“原子写入”方式落盘，避免并发/中断导致文件损坏
# intent: 作为“远端中转器”，文件会被多个请求读写：
#   - 先写入 .tmp，再 os.replace 覆盖，保证读者要么读到旧完整文件，要么读到新完整文件
#   - 自动创建父目录，降低部署出错概率
#   - 不引入文件锁：这里目标是“最后写入者为准”的中转语义，冲突由调用方约束（例如前端只推送当前计划）
def write_json_file_atomic(p: str, obj: Any):
    ensure_dir(os.path.dirname(p))
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
# DOC-END id=manual/relay/json-atomic-write#1

# DOC-BEGIN id=manual/relay/remote-plan-handlers#1 type=logic v=1
# summary: 远端计划中转器：接收当前计划并落盘、读取并返回
# intent: 远端只做“中转存取”：
#   - set：写入带 updated_at/source 的包裹结构，便于另一端判断是否有更新
#   - get：文件不存在时返回空对象而非报错，便于新环境首次启动
async def handle_remote_plan_set(data: RemotePlanSetReq) -> dict:
    payload = {
        "updated_at": time.time(),
        "source": data.source or "",
        "plan": data.plan,
    }
    write_json_file_atomic(RELAY_CURRENT_PLAN_FILE, payload)
    return {"ok": True, "status": "stored", "updated_at": payload["updated_at"]}

async def handle_remote_plan_get(_: RemotePlanGetReq) -> dict:
    if not os.path.exists(RELAY_CURRENT_PLAN_FILE):
        return {"ok": True, "exists": False, "data": {"updated_at": None, "source": "", "plan": []}}

    try:
        with open(RELAY_CURRENT_PLAN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"ok": True, "exists": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read relay plan file: {e}")
# DOC-END id=manual/relay/remote-plan-handlers#1
 
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


# === 任务控制 & LLM 部分 (保持不变) ===

@dataclass
class TaskCtrl:
    stop: threading.Event

TASKS: Dict[str, TaskCtrl] = {}
TASKS_LOCK = threading.Lock()

def ensure_task(task_id: str) -> TaskCtrl:
    with TASKS_LOCK:
        ctrl = TASKS.get(task_id)
        if not ctrl:
            ctrl = TaskCtrl(stop=threading.Event())
            TASKS[task_id] = ctrl
        return ctrl
    
def make_call_tool_with_cancel_detach(should_stop: Callable[[], bool], *, poll: float = 0.05):
    def call_tool_with_cancel(func: Callable[..., Any], args: Dict[str, Any]) -> Tuple[Any, bool]:
        if func is None: return None, should_stop()
        if should_stop(): return None, True

        done = threading.Event()
        out = {"result": None, "err": None}

        def runner():
            try:
                out["result"] = func(**args)
            except Exception as e:
                out["err"] = e
            finally:
                done.set()

        t = threading.Thread(target=runner, daemon=True)
        t.start()

        while True:
            if done.is_set():
                if out["err"] is not None: return out["err"], False
                return out["result"], False
            if should_stop(): return None, True
            time.sleep(poll)
    return call_tool_with_cancel

app = FastAPI()

if os.path.exists(SECRET_FILE):
    with open(SECRET_FILE, "r", encoding="utf-8") as f:
        SERVER_ACCESS_KEY = f.read().strip()
else:
    SERVER_ACCESS_KEY = "default_insecure_password"

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

def run_query_with_tools_safe(fn, result_queue):
    try: fn()
    except Exception: traceback.print_exc()
    finally: result_queue.put(None)

async def handle_llm_simple_query(data: LLMRequestData):
    result_queue = queue.Queue()
    ec = ExternalClient(out_queue=result_queue)
    try:
        llm = LLM(api_key=data.api_key, llm_url=data.llm_url, model_name=data.model_name, format="openai", ec=ec)
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, llm.query, data.question, True)
    except Exception:
        traceback.print_exc()
        result_queue.put(None)
        
    async def event_generator():
        while True:
            item = await loop.run_in_executor(None, result_queue.get)
            if item is None: break
            if isinstance(item, dict):
                c = item.get("data", {}).get("content", "")
                if c: yield c
    return StreamingResponse(event_generator(), media_type="text/plain; charset=utf-8")

async def handle_task_get_terminal_status(data: TaskGetTerminalStatusReq):
    """
    Query all terminal snapshots for a specific task.
    This is used by the frontend for polling updates.
    """
    active_sessions = history_manager.list_task_sessions(data.task_id)
    sessions_data = {}
    
    for sid in active_sessions:
        # Get the real-time screen snapshot for each session
        status = history_manager.get_session_status(data.task_id, sid)
        sessions_data[sid] = {
            "session_id": sid,
            "snapshot": status
        }
        
    pending_cmd = history_manager.get_pending_command(data.task_id)
    
    return {
        "ok": True, 
        "task_id": data.task_id,
        "sessions": sessions_data,
        "pending": pending_cmd
    }
    
async def handle_llm_query(data: LLMRequestData):
    data.task_id = data.task_id or "1"
    ctrl = ensure_task(data.task_id)
    ctrl.stop.clear()
    
    def terminal_create_new() -> str:
        """
        Create a new random terminal session. 
        Returns the generated session_id and initial screen snapshot.
        """
        return history_manager.create_random_session(task_id=data.task_id)

    def terminal_send_command(session_id: str, command: str) -> str:
        """
        LOW-LEVEL TERMINAL INPUT (RAW KEYSTROKES)

        This function sends raw characters directly into a PTY, exactly like
        a human typing on a physical keyboard.

        IMPORTANT SEMANTICS:
        --------------------
        - This function DOES NOT know what a "command" is.
        - It DOES NOT automatically append '\\r' (Enter).
        - It DOES NOT guarantee execution, completion, or correctness.
        - It DOES NOT perform any safety checks.

        Typical Use Cases:
        ------------------
        - Interactive programs (vim, nano, less, htop, python REPL)
        - Password / sudo prompts
        - Sending control sequences (e.g. '\\x03' for Ctrl+C)
        - Navigation keys and incremental input

        DANGER:
        -------
        This is the MOST POWERFUL and MOST DANGEROUS terminal interface.
        Misuse can:
        - Corrupt shell state
        - Interleave inputs
        - Bypass safety mechanisms
        - Cause irreversible side effects

        ONLY use this when you intentionally want to simulate
        low-level human keyboard behavior.

        Parameters:
        -----------
        session_id : str
            Target terminal session identifier.

        command : str
            Raw characters to inject into the PTY.
            To actually execute a shell command, you MUST include '\\r'
            manually at the end of the string.

        Returns:
        --------
        str
            A snapshot of the terminal screen immediately after the input.
            This does NOT imply that execution has completed.
        """
        return history_manager.send_to_session(
            task_id=data.task_id,
            session_id=session_id,
            data=command
        )
        
    def terminal_exec_command(session_id: str, command: str) -> str:
        """
        HIGH-LEVEL SHELL COMMAND EXECUTION (AUTO-ENTER)

        This function represents a higher-level human action:
        typing a complete shell command AND pressing Enter.

        Compared to `terminal_send_command`, this function:
        ---------------------------------------------------
        - Automatically appends '\\r' to the input
        - Intentionally triggers execution
        - Still operates on PTY (not subprocess)
        - Still does NOT guarantee command completion

        What this function IS:
        ----------------------
        - A convenience wrapper for common shell usage
        - A semantic signal: "this is intended to be executed"
        - Safer than raw typing, but still NOT fully safe

        What this function is NOT:
        --------------------------
        - It does NOT wait for the command to finish
        - It does NOT parse output
        - It does NOT detect shell prompt
        - It does NOT protect against dangerous commands

        Appropriate Use Cases:
        ----------------------
        - Simple shell commands (ls, cd, cat, pwd)
        - Non-interactive utilities
        - LLM tool calls that represent a single command

        NOT suitable for:
        -----------------
        - vim / nano / less
        - Password input
        - Multi-step interactive workflows

        Parameters:
        -----------
        session_id : str
            Target terminal session identifier.

        command : str
            Shell command WITHOUT trailing '\\r'.

        Returns:
        --------
        str
            A snapshot of the terminal screen immediately after
            the Enter key is pressed.
            Execution may still be in progress.
        """
        # Normalize input: remove accidental newlines
        cmd = command.rstrip("\n\r")

        # Append Enter explicitly
        return history_manager.send_to_session(
            task_id=data.task_id,
            session_id=session_id,
            data=cmd + "\r"
        )



    def terminal_get_status(session_id: str) -> str:
        """
        Get the current screen content of a session without sending any commands.
        """
        return history_manager.get_session_status(
            task_id=data.task_id, 
            session_id=session_id
        )

    def terminal_terminate_session(session_id: str) -> str:
        """
        Terminate and destroy a terminal session to release system resources.
        """
        return history_manager.close_terminal(
            task_id=data.task_id, 
            session_id=session_id
        )
        
    def terminal_stage_command(session_id: str, command: str, reason: str) -> str:
        """
        If a command is potentially dangerous (e.g., rm, pip install, editing config), 
        use this tool to stage it for human review instead of executing it directly.
        """
        return history_manager.stage_unsafe_command(
            task_id=data.task_id, 
            session_id=session_id, 
            command=command, 
            reason=reason
        )
    
    task_dir=f"{DATA_DIR}/tasks/{data.task_id}"
    tc = Tool_Calls(LOG_DIR=task_dir, MAX_CHAR=800000, mode="Summary")
    tc.remove_temporary_context()
    if data.pinned_codes:
        tc.inject_temporary_context(data.pinned_codes)
    result_queue = queue.Queue()
    ec = ExternalClient(out_queue=result_queue)
    tools = []
    ct = custom_tools()
    if data.enable_fc:
        tools.extend([
            ct.func_search,
            terminal_create_new, 
            terminal_send_command, 
            terminal_exec_command,
            # terminal_get_status, 
            terminal_terminate_session,
            terminal_stage_command,
        ])
    
    def system_prompt() -> str:
        # 实时拉取当前任务下的所有会话
        active_sessions = history_manager.list_task_sessions(data.task_id)
        
        terminal_context_block = "\n=== ACTIVE TERMINAL SESSIONS ===\n"
        if not active_sessions:
            terminal_context_block += "No active terminal sessions in this task.\n"
        else:
            for sid in active_sessions:
                # 获取该会话最新的实时状态（屏幕回显）
                status = history_manager.get_session_status(data.task_id, sid)
                terminal_context_block += f"--- Session ID: {sid} ---\n{status}\n"
        terminal_context_block += "================================\n"
        
        # 基础规则
        base_rules = [
            "1. Answer in the same language.",
            "2. Use $$ for math (e.g. $$x^2$$).",
            "3. Pinyin -> Chinese.",
            "4. If clarification is needed, ask the user directly without invoking any tools.",
            (
                "5. For newly generated code (do NOT add comments to existing legacy code unless explicitly requested), "
                "you MUST provide detailed Chinese comments using a unified block format with stable IDs. "
                "Use BEGIN/END markers and include at least 'summary' and 'intent' fields.\n"
                "Required structure (Python example using '#'):\n"
                "# DOC-BEGIN id=<stable_id> type=<kind> v=<n>\n"
                "# summary: <一句话说明这段代码做什么>\n"
                "# intent: <为什么这样写/约束/取舍/风险/边界条件>\n"
                "# DOC-END id=<stable_id>\n"
                "Rules:\n"
                "- The 'id' must be unique within the file and stable over time; do NOT use line numbers.\n"
                "- 'DOC-END' must repeat the same id to prevent mismatch.\n"
                "- DOC-BEGIN/DOC-END 必须严格包裹你要注释的那段代码：DOC-BEGIN 放在代码段第一行之前，DOC-END 放在该代码段最后一行之后（而不是紧跟在注释字段后面）。\n"
                "- 不是说“一段代码只写一个注释块”；而是任何你（LLM）判断需要解释的局部（非直观逻辑/约束/取舍/边界/安全/性能/外部依赖假设）都应在其对应代码段附近增加一个 DOC 块。\n"
                "- Place DOC-BEGIN immediately before the commented code block and DOC-END immediately after it.\n"
                "- The comment content MUST be in Chinese and sufficiently detailed (focus on why/constraints/edge cases; avoid merely restating the code).\n"
                "Example (add 函数做两个注释：函数整体 + 函数内某一行；下面的 a*b 函数不做注释):\n"
                "```python\n"
                "# DOC-BEGIN id=example/math/add#1 type=design v=1\n"
                "# summary: 计算 a 与 b 的和，并定义缺省值策略\n"
                "# intent: 演示“注释块必须包裹代码段”的位置规则；同时说明当存在边界条件策略（如 None 处理）时，应对局部决策单独加注释，避免把关键约定埋在实现细节里\n"
                "def add(a, b):\n"
                "    # DOC-BEGIN id=example/math/add/none-as-zero#1 type=behavior v=1\n"
                "    # summary: 将 None 视为 0 的输入归一化策略\n"
                "    # intent: 明确边界条件与业务约定；该策略可能掩盖上游缺失值问题，因此需要显式记录，且仅在确认为期望行为时使用\n"
                "    if a is None: a = 0\n"
                "    # DOC-END id=example/math/add/none-as-zero#1\n"
                "    if b is None: b = 0\n"
                "    return a + b\n"
                "# DOC-END id=example/math/add#1\n"
                "\n"
                "def mul(a, b):\n"
                "    return a * b\n"
                "```\n"
            )
        ]
        
        code_edit_protocol = ""
        if data.system_prompt_mode == "concise":
            code_edit_protocol = (
                "\n=== CODE EDIT PROTOCOL ===\n"
                "You can propose edits to PINNED code blocks (identified by Hash) using the following format. "
                "DO NOT output the full file; ONLY provide the patch using this EXACT structure:\n\n"
                "§ Start (hash_value)\n"
                "[precise lines of existing code where the edit begins, must be uniquely matched]\n"
                "§ Start (hash_value)\n\n"
                "§ End (hash_value)\n"
                "[precise lines of existing code where the edit ends, must be uniquely matched]\n"
                "§ End (hash_value)\n\n"
                "§ Replace (hash_value)\n"
                "[The new code that replaces everything between and including the Start/End anchors]\n"
                "§ Replace (hash_value)\n\n"
                "RULES FOR EDITING:\n"
                "- The text inside 'Start' and 'End' MUST be copied verbatim from the pinned file and matched by strict character-for-character comparison (including whitespace and blank lines).\n"
                "- Each anchor snippet MUST occur exactly once in the pinned file (occurrence count == 1). If not, DO NOT produce a patch; ask for more surrounding lines to make anchors unique.\n"
                "- You will find the current content of pinned files in the [TEMPORARY_CONTEXT] assistant message provided in the history.\n"
                "- Refer to files ONLY by their Hash value.\n"
                "EXAMPLE:\n\n"
                "Suppose the pinned file (hash = abc123) contains the following code:\n\n"
                "```python\n"
                "def add(a, b):\n"
                "    return a + b\n\n"
                "```\n"
                "To modify this function to handle None values, you should output:\n\n"
                "§ Start (abc123)\n"
                "```python\n"
                "def add(a, b):\n"
                "```\n"
                "§ Start (abc123)\n\n"
                "§ End (abc123)\n"
                "```python\n"
                "    return a + b\n"
                "```\n"
                "§ End (abc123)\n\n"
                "§ Replace (abc123)\n"
                "```python\n"
                "def add(a, b):\n"
                "    if a is None or b is None:\n"
                "        return None\n"
                "    return a + b\n"
                "```\n"
                "§ Replace (abc123)\n"
            )

        if data.system_prompt_mode in ["concise"]:
            base_rules.append("5. Be concise.")
            
        # 获取挂起状态并注入 System Prompt
        pending = history_manager.get_pending_command(data.task_id)
        pending_context = ""
        if pending:
            pending_context = (
                f"\n[!!! PENDING APPROVAL !!!]\n"
                f"There is a command waiting for user execution:\n"
                f"Session: {pending['session_id']}\n"
                f"Command: {pending['command']}\n"
                f"Reason: {pending['reason']}\n"
                f"The user is reviewing this. DO NOT suggest new destructive commands until this is processed.\n"
            )
            
        return "\n".join(base_rules) + "\n" + terminal_context_block + "\n" + code_edit_protocol + "\n" + pending_context

    try:
        llm = LLM(api_key=data.api_key, llm_url=data.llm_url, model_name=data.model_name, format="openai", ec=ec)
        loop = asyncio.get_running_loop()
        fn = partial(
            llm.query_with_tools,
            system_prompt=system_prompt,
            prompt=data.question,
            max_steps=100,
            tc=tc,
            tools=tools,
            verbose=True,
            stop_condition=stop_if_no_tool_calls,
            tool_runner=make_call_tool_with_cancel_detach(lambda: ctrl.stop.is_set()),
        )
        loop.run_in_executor(None, run_query_with_tools_safe, fn, result_queue)
    except Exception:
        result_queue.put(None)
        traceback.print_exc()
        
    async def event_generator():
        while True:
            item = await loop.run_in_executor(None, result_queue.get)
            if item is None: break
            if isinstance(item, dict):
                c = item.get("data", {}).get("content", "")
                if c: yield c
    return StreamingResponse(event_generator(), media_type="text/plain; charset=utf-8")

# === Router ===

async def verify_server_access_key(x_server_api_key: str = Header(None)):
    if x_server_api_key != SERVER_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid Key")
    return x_server_api_key

@app.post("/gateway", dependencies=[Depends(verify_server_access_key)])
async def gateway_endpoint(req: ActionRequest):
    print(f"Action: {req.action}")
    print(req)
    try:
        req.data = with_default_playbook_root(req.action, req.data)
        if req.action == "llm_simple_query":
            return await handle_llm_simple_query(TypeAdapter(LLMRequestData).validate_python(req.data))
        if req.action == "llm_query":
            return await handle_llm_query(TypeAdapter(LLMRequestData).validate_python(req.data))
        if req.action == "llm_stop_task":
            return {"ok": True}
        
        # Article Ops
        if req.action == "article_list":
            return await handle_article_list(TypeAdapter(ArticleListReq).validate_python(req.data))
        if req.action == "article_get_html":
            return await handle_article_get_html(TypeAdapter(ArticleGetHtmlReq).validate_python(req.data))
        if req.action == "task_select_article":
            return await handle_task_select_article(TypeAdapter(TaskSelectArticleReq).validate_python(req.data))
        if req.action == "article_extract_images":
            return await handle_article_extract_images(TypeAdapter(ArticleExtractImagesReq).validate_python(req.data))
        
        if req.action == "task_get_terminal_status":
            return await handle_task_get_terminal_status(
                TypeAdapter(TaskGetTerminalStatusReq).validate_python(req.data)
            )
        if req.action == "task_execute_pending":
            task_id = req.data.get("task_id")
            pending = history_manager.pop_pending_command(task_id)
            if not pending:
                return {"ok": False, "msg": "No pending command."}
            
            # 真正执行
            result = history_manager.send_to_session(
                task_id=task_id,
                session_id=pending["session_id"],
                data=pending["command"] + "\r"
            )
            
            # 在 tc (任务历史) 中记录这一手动操作，保持 LLM 知情
            task_dir = f"{DATA_DIR}/tasks/{task_id}"
            tc = Tool_Calls(LOG_DIR=task_dir, MAX_CHAR=800000, mode="Summary")
            tc.extend([{
                "role": "assistant",
                "content": f"【User Manual Action】User approved and executed: {pending['command']}\nResult:\n{result}"
            }])
            
            return {"ok": True, "result": result}

        # 清除/拒绝挂起的命令
        if req.action == "task_discard_pending":
            task_id = req.data.get("task_id")
            pending = history_manager.pop_pending_command(task_id)
            
            # 告知 LLM 用户拒绝了
            task_dir = f"{DATA_DIR}/tasks/{task_id}"
            tc = Tool_Calls(LOG_DIR=task_dir, MAX_CHAR=800000, mode="Summary")
            tc.extend([{
                "role": "assistant",
                "content": f"【User Manual Action】User REJECTED the pending command: {pending['command']}."
            }])
            return {"ok": True}
        
        if req.action == "article_generate_blog":
            return await handle_article_generate_blog(
                TypeAdapter(ArticleGenerateBlogReq).validate_python(req.data)
            )
            
        if req.action == "article_delete_blog":
            return await handle_article_delete_blog(
                TypeAdapter(ArticleDeleteBlogReq).validate_python(req.data)
            )

        # Relay (Remote Plan)
        if req.action == "remote_plan_set":
            return await handle_remote_plan_set(
                TypeAdapter(RemotePlanSetReq).validate_python(req.data)
            )
        if req.action == "remote_plan_get":
            return await handle_remote_plan_get(
                TypeAdapter(RemotePlanGetReq).validate_python(req.data)
            )
            
            
            
        # Playbook Ops
        if req.action == "playbook_list":
            return await handle_playbook_list(
                TypeAdapter(PlaybookListReq).validate_python(req.data)
            )

        if req.action == "playbook_get":
            return await handle_playbook_get(
                TypeAdapter(PlaybookGetReq).validate_python(req.data)
            )

        if req.action == "playbook_upsert_policy":
            return await handle_playbook_upsert_policy(
                TypeAdapter(PlaybookUpsertPolicyReq).validate_python(req.data)
            )

        if req.action == "playbook_upsert_backlog":
            return await handle_playbook_upsert_backlog(
                TypeAdapter(PlaybookUpsertBacklogReq).validate_python(req.data)
            )

        if req.action == "playbook_upsert_assumption":
            return await handle_playbook_upsert_assumption(
                TypeAdapter(PlaybookUpsertAssumptionReq).validate_python(req.data)
            )

        if req.action == "playbook_delete":
            return await handle_playbook_delete(
                TypeAdapter(PlaybookDeleteReq).validate_python(req.data)
            )
            
        # Playbook - Suggest next step via LLM
        if req.action == "playbook_suggest_next_step":
            return await handle_playbook_suggest_next_step(
                TypeAdapter(PlaybookSuggestNextStepReq).validate_python(req.data)
            )
            
        if req.action == "task_dehydrate_code":
            task_id = req.data.get("task_id")
            content = req.data.get("content")
            hash_val = req.data.get("hash")
            filename = req.data.get("filename", "code.py")
            
            task_dir = f"{DATA_DIR}/tasks/{task_id}"
            tc = Tool_Calls(LOG_DIR=task_dir, MAX_CHAR=800000)
            
            # 执行脱水
            tc.dehydrate_content(content, hash_val, filename)
            # 同时注入一次“挂起确认”，让 LLM 知道它现在可以用这个 Hash 了
            tc.inject_pinned_content(hash_val, filename, content)
            
            return {"ok": True}
        
        if req.action == "task_reinject_code":
            task_id = req.data.get("task_id")
            content = req.data.get("content")
            hash_val = req.data.get("hash")
            filename = req.data.get("filename")
            
            task_dir = f"{DATA_DIR}/tasks/{task_id}"
            tc = Tool_Calls(LOG_DIR=task_dir, MAX_CHAR=800000)
            tc.inject_pinned_content(hash_val, filename, content)
            
            return {"ok": True}

        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)