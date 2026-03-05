# server.py
import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends 
from fastapi.middleware.cors import CORSMiddleware
import os
from src.services.agents import Tool_Calls
import traceback
from pydantic import TypeAdapter
from src.services.session_history import history_manager
import logging

from src.utils.playbook_store import (
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

from src.models.requests import (
    PlaybookSuggestNextStepReq, RemotePlanSetReq, RemotePlanGetReq,
    ArticleDeleteBlogReq, TaskGetTerminalStatusReq, ArticleExtractImagesReq,
    ArticleListReq, ArticleGetHtmlReq,
    TaskSelectArticleReq, LLMRequestData, ActionRequest,
    ArticleGenerateBlogReq,
)
from src.config import (
    SECRET_FILE, DATA_DIR, with_default_playbook_root,
)

from src.handlers.article_handlers import (
    handle_article_get_html, handle_article_list,
    handle_article_delete_blog, handle_article_extract_images,
    handle_task_select_article,
    handle_article_generate_blog,
)

from src.handlers.llm_handlers import (
    handle_llm_simple_query, handle_llm_query, handle_task_get_terminal_status,
)

from src.handlers.relay_handlers import handle_remote_plan_set, handle_remote_plan_get
from src.handlers.playbook_handlers import handle_playbook_suggest_next_step

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)

# === 任务控制 & LLM 部分 (保持不变) ===
app = FastAPI()

if os.path.exists(SECRET_FILE):
    with open(SECRET_FILE, "r", encoding="utf-8") as f:
        SERVER_ACCESS_KEY = f.read().strip()
else:
    SERVER_ACCESS_KEY = "default_insecure_password"

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
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