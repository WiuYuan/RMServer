import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.services.llm import LLM 
import os
import asyncio
import queue
from src.services.external_client import ExternalClient
from fastapi.responses import StreamingResponse
from src.services.agents import Tool_Calls, stop_if_no_tool_calls
import traceback
from functools import partial
from src.services.custom_tools import custom_tools
import threading, time
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any, Optional, Union, Literal
from pydantic import TypeAdapter
import shutil, json, re
from bs4 import BeautifulSoup
import trafilatura

# === 基础配置 ===
SECRET_FILE = "/home/ubuntu/.env_secret"
DATA_DIR = "/home/ubuntu/workspace/data"
ARTICLES_ROOT = f"{DATA_DIR}/articles"

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

class LLMRequestData(BaseModel):
    model_name: str
    question: str
    api_key: str
    llm_url: Optional[str] = "https://api.deepseek.com/v1/chat/completions"
    task_id: Optional[str] = "1"
    system_prompt_mode: Optional[str] = "default"
    enable_fc: Optional[bool] = False

class ActionRequest(BaseModel):
    action: str
    data: dict

class StopData(BaseModel):
    task_id: str

# === 核心辅助函数 ===

def load_raw_html(abs_path: str) -> Tuple[str, str]:
    filename = os.path.basename(abs_path)
    title = filename[:-5] if filename.lower().endswith(".html") else filename
    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    return title, html

async def handle_article_get_html(data: ArticleGetHtmlReq):
    abs_path = resolve_article_abs_path(data.article_id)
    title, html = load_raw_html(abs_path)
    return {"ok": True, "article_id": data.article_id, "title": title, "clean_html": html}
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
                f"【System Context Injection】\n"
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

# === 业务 Handlers ===
async def handle_article_extract_images(data: ArticleExtractImagesReq):
    abs_path = resolve_article_abs_path(data.article_id)
    
    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "lxml")

    images_data = []
    # 查找所有 img 标签，BeautifulSoup 会按 DOM 顺序返回
    img_tags = soup.find_all("img")
    
    count = 1
    for img in img_tags:
        src = img.get("src", "")
        if src.startswith("data:image"):
            # 检查格式是否为 jpg 或 webp
            # 格式示例: data:image/jpeg;base64,/9j/4AAQ...
            header_part = src.split(",")[0]
            
            ext = None
            if "image/jpeg" in header_part or "image/jpg" in header_part:
                ext = "jpg"
            elif "image/webp" in header_part:
                ext = "webp"
            
            if ext:
                # 获取 alt 属性作为描述，如果没有则生成默认标题
                alt_text = img.get("alt", "").strip()
                caption = alt_text if alt_text else f"Figure {count}"
                
                images_data.append({
                    "index": count,
                    "filename": f"figure_{count}.{ext}",
                    "caption": caption,
                    "mime_type": "image/jpeg" if ext == "jpg" else "image/webp",
                    "base64_content": src # 包含完整的 data:image... 前缀，前端可直接展示
                })
                count += 1

    return {
        "ok": True,
        "article_id": data.article_id,
        "total": len(images_data),
        "images": images_data
    }

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
    
async def handle_llm_query(data: LLMRequestData):
    data.task_id = data.task_id or "1"
    ctrl = ensure_task(data.task_id)
    ctrl.stop.clear()
    
    task_dir=f"{DATA_DIR}/tasks/{data.task_id}"
    tc = Tool_Calls(LOG_DIR=task_dir, MAX_CHAR=800000, mode="Summary")
    result_queue = queue.Queue()
    ec = ExternalClient(out_queue=result_queue)
    tools = []
    ct = custom_tools()
    if data.enable_fc:
        tools.extend([ct.web_fetch, ct.search])
        
    system_prompt = "1. Answer in the same language.\n2. Use $$ for math.\n3. Pinyin -> Chinese.\n"
    if data.system_prompt_mode == "concise":
        system_prompt += "4. Be concise.\n"

    try:
        llm = LLM(api_key=data.api_key, llm_url=data.llm_url, model_name=data.model_name, format="openai", ec=ec)
        loop = asyncio.get_running_loop()
        fn = partial(
            llm.query_with_tools,
            prompt=system_prompt + "\n" + data.question,
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
    try:
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
            
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)