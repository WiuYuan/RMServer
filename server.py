# server.py
import uvicorn
import json as _json
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
from src.services.task_manager import stop_task

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
        # DOC-BEGIN id=server/llm_stop_task#1 type=api v=1
        # summary: llm_stop_task 端点——设置 ctrl.stop 信号并通过 asyncio.to_thread 等待 LLM 线程退出，
        #   返回 finished 标识是否在超时内成功停止
        # intent: stop_task 内部会 ctrl.done.wait(timeout=15) 阻塞，
        #   必须用 to_thread 卸载到线程池，否则会冻结 FastAPI 的 async event loop。
        #   finished=False 表示超时（线程可能卡在长时间 tool call 中），前端仍可认为"已请求停止"。
        if req.action == "llm_stop_task":
            task_id = req.data.get("task_id")
            if not task_id:
                return {"ok": False, "error": "task_id required"}
            import asyncio
            finished = await asyncio.to_thread(stop_task, task_id)
            return {"ok": True, "finished": finished}
        # DOC-END id=server/llm_stop_task#1
        
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

        if req.action == "terminal_get_status":
            session_id = req.data.get("session_id")
            task_id = req.data.get("task_id", "")
            status = history_manager.get_session_status(task_id, session_id)
            return {"ok": True, "session_id": session_id, "snapshot": status}

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

        # === Global Terminal Operations ===
        # DOC-BEGIN id=server/terminal-global-ops#1 type=api v=2
        # summary: 全局 terminal 管理——创建、关闭、列表、挂载/卸载、发送命令
        # intent: Terminal 不再属于 task，而是全局独立资源。前端 Terminal 面板统一管理。
        #   创建时可选 auto_mount_task 自动挂载到指定 task。
        #   关闭是全局操作，自动清理所有 task 的引用。
        #   exec/send 保留 task_id 参数用于未来权限校验。
        if req.action == "terminal_create":
            task_id = req.data.get("task_id")
            result_str = history_manager.create_terminal(auto_mount_task=task_id)
            result_obj = _json.loads(result_str)
            return {"ok": True, "session_id": result_obj["session_id"], "detail": result_obj}

        if req.action == "terminal_close":
            session_id = req.data.get("session_id")
            result = _json.loads(history_manager.close_terminal(session_id))
            return result

        if req.action == "terminal_list_all":
            terminals = history_manager.list_all_terminals()
            return {"ok": True, "terminals": terminals}

        if req.action == "terminal_mount":
            task_id = req.data.get("task_id")
            session_id = req.data.get("session_id")
            result = _json.loads(history_manager.mount_terminal(task_id, session_id))
            return result

        if req.action == "terminal_unmount":
            task_id = req.data.get("task_id")
            session_id = req.data.get("session_id")
            result = _json.loads(history_manager.unmount_terminal(task_id, session_id))
            return result

        # DOC-BEGIN id=server/terminal-exec-send#1 type=behavior v=1
        # summary: terminal_exec 和 terminal_send 增加错误检测，send_to_session 返回 "Error:" 开头时标记 ok=False
        # intent: send_to_session 在 terminal 不存在时返回错误字符串，但之前 server 层不检查，
        #   一律返回 ok=True，导致前端/调用方误以为操作成功。现在通过前缀匹配检测错误。
        if req.action == "terminal_exec":
            task_id = req.data.get("task_id")
            session_id = req.data.get("session_id")
            command = req.data.get("command", "")
            cmd = command.rstrip("\n\r")
            result = history_manager.send_to_session(task_id, session_id, cmd + "\r")
            if isinstance(result, str) and result.startswith("Error:"):
                return {"ok": False, "error": result}
            return {"ok": True, "result": result}

        if req.action == "terminal_send":
            task_id = req.data.get("task_id")
            session_id = req.data.get("session_id")
            command = req.data.get("command", "")
            result = history_manager.send_to_session(task_id, session_id, command)
            if isinstance(result, str) and result.startswith("Error:"):
                return {"ok": False, "error": result}
            return {"ok": True, "result": result}
        # DOC-END id=server/terminal-exec-send#1
        # DOC-END id=server/terminal-global-ops#1

        # === Workspace Management (一等公民) ===
        # DOC-BEGIN id=server/workspace-ops#1 type=api v=1
        # summary: Workspace CRUD + Task 关联/解除——创建、删除、列表、关联、解除、查询状态、toggle bound
        # intent: Workspace 是独立实体（包含 1 个 bound terminal + 1 个 interactive terminal）。
        #   Task 通过 attach 关联 workspace 获得文件操作能力。所有操作走统一 gateway。
        if req.action == "workspace_create":
            result = _json.loads(history_manager.create_workspace())
            return result

        # DOC-BEGIN id=server/workspace-create-from-recording#1 type=api v=1
        # summary: workspace_create_from_recording 端点——传入 recording_id 和可选 speed_factor，
        #   后端一站式完成创建双 terminal + replay + binding，返回 workspace_id 和初始状态
        # intent: 将前端多步编排下沉为单次 API 调用，前端通过 workspace_info 查看进度。
        if req.action == "workspace_create_from_recording":
            recording_id = req.data.get("recording_id")
            speed_factor = req.data.get("speed_factor", 1.0)
            result = _json.loads(history_manager.workspace_create_from_recording(recording_id, speed_factor))
            return result
        # DOC-END id=server/workspace-create-from-recording#1

        if req.action == "workspace_delete":
            ws_id = req.data.get("workspace_id")
            result = _json.loads(history_manager.delete_workspace(ws_id))
            return result

        if req.action == "workspace_list":
            workspaces = history_manager.list_workspaces()
            return {"ok": True, "workspaces": workspaces}

        if req.action == "workspace_attach":
            task_id = req.data.get("task_id")
            ws_id = req.data.get("workspace_id")
            result = _json.loads(history_manager.attach_workspace(task_id, ws_id))
            return result

        if req.action == "workspace_detach":
            task_id = req.data.get("task_id")
            result = _json.loads(history_manager.detach_workspace(task_id))
            return result

        if req.action == "workspace_info":
            task_id = req.data.get("task_id")
            ws_id = req.data.get("workspace_id")
            info = history_manager.get_workspace_info(task_id=task_id, workspace_id=ws_id)
            return {"ok": True, "info": info}

        if req.action == "terminal_toggle_bound":
            session_id = req.data.get("session_id")
            result = _json.loads(history_manager.toggle_terminal_bound(session_id))
            return result
        # DOC-END id=server/workspace-ops#1

        # === Recording & Replay ===
        # DOC-BEGIN id=server/recording-ops#1 type=api v=1
        # summary: 命令录制与回放 API——获取/清空命令日志、保存/列出/删除录制、回放/停止回放
        # intent: 完整的录制生命周期管理。terminal 录制保存单个 terminal 的命令序列，
        #   workspace 录制保存两个 terminal 的命令序列。回放支持指定速度因子和目标 terminal。
        if req.action == "recording_get_log":
            session_id = req.data.get("session_id")
            return history_manager.get_terminal_command_log(session_id)

        if req.action == "recording_clear_log":
            session_id = req.data.get("session_id")
            return history_manager.clear_terminal_command_log(session_id)

        if req.action == "recording_save_terminal":
            session_id = req.data.get("session_id")
            name = req.data.get("name")
            return history_manager.save_terminal_recording(session_id, name)

        if req.action == "recording_save_workspace":
            workspace_id = req.data.get("workspace_id")
            name = req.data.get("name")
            return history_manager.save_workspace_recording(workspace_id, name)

        if req.action == "recording_list":
            rec_type = req.data.get("type")
            return history_manager.list_recordings(rec_type)

        if req.action == "recording_get":
            recording_id = req.data.get("recording_id")
            return history_manager.get_recording(recording_id)

        if req.action == "recording_delete":
            recording_id = req.data.get("recording_id")
            return history_manager.delete_recording(recording_id)

        # DOC-BEGIN id=server/recording-replay-async#1 type=behavior v=1
        # summary: recording_replay 端点在 wait=True 时通过 asyncio.to_thread 将阻塞的 thread.join 
        #   放到线程池执行，避免卡死 FastAPI 的 async event loop
        # intent: replay_recording(wait=True) 内部会 thread.join() 阻塞数秒到数分钟，
        #   如果直接在 async def 中调用会冻结整个服务器。to_thread 将其卸载到默认线程池。
        #   wait=False 时无阻塞，直接调用即可。
        if req.action == "recording_replay":
            recording_id = req.data.get("recording_id")
            target_session_id = req.data.get("target_session_id")
            target_bound_tid = req.data.get("target_bound_tid")
            target_inter_tid = req.data.get("target_inter_tid")
            speed_factor = req.data.get("speed_factor", 1.0)
            wait = req.data.get("wait", False)
            import asyncio
            if wait:
                return await asyncio.to_thread(
                    history_manager.replay_recording,
                    recording_id,
                    target_session_id=target_session_id,
                    target_bound_tid=target_bound_tid,
                    target_inter_tid=target_inter_tid,
                    speed_factor=speed_factor,
                    wait=True,
                )
            else:
                return history_manager.replay_recording(
                    recording_id,
                    target_session_id=target_session_id,
                    target_bound_tid=target_bound_tid,
                    target_inter_tid=target_inter_tid,
                    speed_factor=speed_factor,
                    wait=False,
                )
        # DOC-END id=server/recording-replay-async#1

        if req.action == "recording_replay_stop":
            session_id = req.data.get("session_id")
            return history_manager.stop_replay(session_id)

        if req.action == "recording_replay_status":
            return history_manager.get_replay_status()
        # DOC-END id=server/recording-ops#1

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
    uvicorn.run(app, host="0.0.0.0", port=8888)