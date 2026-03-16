# src/handlers/llm_handlers.py
import os
import asyncio
import json
import queue
import traceback
from functools import partial

from fastapi.responses import StreamingResponse

from src.config import DATA_DIR
from src.services.llm import LLM
from src.services.external_client import ExternalClient
from src.services.agents import Tool_Calls, stop_if_no_tool_calls
from src.services.custom_tools import custom_tools
from src.services.session_history import history_manager
from src.services.task_manager import ensure_task, make_call_tool_with_cancel_detach, run_query_with_tools_safe
from src.models.requests import LLMRequestData, TaskGetTerminalStatusReq


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
        
    # DOC-BEGIN id=llm_handlers/event_generator#1 type=core v=1
    # summary: SSE 事件生成器——从 result_queue 逐条取出消息，区分纯文本 (type=llm_info) 和
    #   结构化事件 (type=image 等)，分别以不同前缀 yield 给前端
    # intent: 原有协议只输出纯文本（content 字符串直接 yield），无法传递图片等二进制数据。
    #   扩展后约定：type="image" 等非文本事件以 "§IMG§" + JSON 的格式 yield，
    #   前端通过检测此前缀区分文本和结构化事件。选择内联标记而非切换 media_type，
    #   是因为 SSE 流中文本和图片事件交替出现，不能中途改 Content-Type。
    #   "§IMG§" 前缀足够独特，不会与正常 LLM 输出冲突。
    async def event_generator():
        while True:
            item = await loop.run_in_executor(None, result_queue.get)
            if item is None:
                break
            if isinstance(item, dict):
                msg_type = item.get("type", "")
                if msg_type == "image":
                    yield "§IMG§" + json.dumps(item["data"], ensure_ascii=False)
                else:
                    c = item.get("data", {}).get("content", "")
                    if c:
                        yield c
    return StreamingResponse(event_generator(), media_type="text/plain; charset=utf-8")
    # DOC-END id=llm_handlers/event_generator#1

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

    result_queue = queue.Queue()
    ec = ExternalClient(out_queue=result_queue)
    
    def terminal_create_new() -> str:
        """
        Create a new random terminal session. 
        Returns the generated session_id and initial screen snapshot.
        """
        return history_manager.create_terminal(auto_mount_task=data.task_id)

    # DOC-BEGIN id=llm_handlers/terminal_send_command#1 type=tool v=3
    # summary: 低层终端输入工具——接收 session_id、command、wait_seconds 三个参数，
    #   将原始字符注入 PTY（不追加回车），sleep wait_seconds 秒等待执行，仅返回提交确认
    # intent: 终端屏幕状态已通过 system_prompt 在每个 step 开始时自动注入上下文，
    #   tool call 返回值中不再需要携带快照，避免重复占用 token。
    #   wait_seconds 与 terminal_exec_command 保持一致的设计——PTY 是异步的，
    #   交互式程序（vim、python REPL 等）同样需要等待响应后再进入下一个 step。
    #   默认 0.5 秒（比 exec 的 2 秒短，因为 raw input 通常是单次按键/短序列）。
    #   上限 30 秒防止阻塞过久。
    def terminal_send_command(session_id: str, command: str, wait_seconds: float = 0.5) -> str:
        """
        Parameters:
        -----------
        session_id : str
            Target terminal session identifier.

        command : str
            Raw characters to inject into the PTY.
            To actually execute a shell command, you MUST include '\\r'
            manually at the end of the string.

        wait_seconds : float
            Number of seconds to wait after sending the raw input before returning.
            This gives interactive programs time to respond so the next step's context
            injection captures the updated screen.
            Default is 0.5 seconds. Maximum is 30 seconds.

        Returns:
        --------
        str
            A confirmation message. Terminal screen content is available
            in the system prompt context at the next step.
        """
        import time

        wait_seconds = max(0.0, min(float(wait_seconds), 30.0))

        result = history_manager.send_to_session(
            task_id=data.task_id,
            session_id=session_id,
            data=command
        )
        if result.startswith("Error:"):
            return result

        if wait_seconds > 0:
            time.sleep(wait_seconds)

        return f"OK: raw input sent to session '{session_id}', waited {wait_seconds}s. Screen will refresh in next step context."
    # DOC-END id=llm_handlers/terminal_send_command#1
        
    # DOC-BEGIN id=llm_handlers/terminal_exec_command#1 type=tool v=2
    # summary: 高层终端命令执行工具——接收 session_id、command、wait_seconds 三个参数，
    #   自动追加回车发送命令，sleep wait_seconds 秒等待执行，仅返回提交确认而非屏幕快照
    # intent: terminal PTY 是异步的——send_to_session 只是将字符注入 PTY 输入流，
    #   命令的实际执行需要时间（编译、网络请求、文件 IO 等）。wait_seconds 让命令有时间执行完毕，
    #   默认 2 秒覆盖大部分简单命令；对于耗时操作 LLM 应主动传入更大的值。上限 30 秒防止阻塞过久。
    #   不再返回屏幕快照——终端状态已通过 system_prompt 在每个 step 开始时自动注入上下文，
    #   tool call 返回快照会与上下文注入重复，浪费 token。
    def terminal_exec_command(session_id: str, command: str, wait_seconds: float = 2.0) -> str:
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

        wait_seconds : float
            Number of seconds to wait after sending the command before returning.
            This gives the command time to execute so the next step's context
            injection captures the completed output.
            Default is 2.0 seconds. Maximum is 30 seconds.
            Use larger values for slow commands (e.g. pip install, compilation).

        Returns:
        --------
        str
            A confirmation message. Terminal screen content is available
            in the system prompt context at the next step.
        """
        import time

        cmd = command.rstrip("\n\r")
        wait_seconds = max(0.0, min(float(wait_seconds), 30.0))

        result = history_manager.send_to_session(
            task_id=data.task_id,
            session_id=session_id,
            data=cmd + "\r"
        )
        if result.startswith("Error:"):
            return result

        if wait_seconds > 0:
            time.sleep(wait_seconds)

        return f"OK: command submitted to session '{session_id}', waited {wait_seconds}s. Screen will refresh in next step context."
    # DOC-END id=llm_handlers/terminal_exec_command#1



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
        return history_manager.close_terminal(session_id=session_id)
        
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

    # === Remote File Operations (via bound workspace terminal) ===

    def remote_read_file(path: str) -> str:
        """
        Read a file from the remote server via the bound workspace terminal.
        The file is read in chunks to handle large files (>1000 lines).
        Returns the full file content as a string.
        
        Parameters:
        -----------
        path : str
            Absolute path on the remote server, or relative path from the bound work_dir.
        """
        result = history_manager.remote_read_file(task_id=data.task_id, path=path)
        if result["ok"]:
            return result["content"]
        return json.dumps(result, ensure_ascii=False)

    def remote_write_file(path: str, content: str) -> str:
        """
        Write content to a file on the remote server via the bound workspace terminal.
        Overwrites the file if it exists, creates it if it doesn't.
        
        Parameters:
        -----------
        path : str
            Absolute path on the remote server, or relative path from the bound work_dir.
        content : str
            The full file content to write.
        """
        result = history_manager.remote_write_file(task_id=data.task_id, path=path, content=content)
        return json.dumps(result, ensure_ascii=False)

    def remote_list_dir(path: str) -> str:
        """
        List files and directories at the given path on the remote server.
        Returns ls -la output.
        
        Parameters:
        -----------
        path : str
            Absolute path on the remote server, or relative path from the bound work_dir.
        """
        result = history_manager.remote_list_dir(task_id=data.task_id, path=path)
        if result["ok"]:
            return result["stdout"]
        return json.dumps(result, ensure_ascii=False)

    def remote_file_exists(path: str) -> str:
        """
        Check if a file exists at the given path on the remote server.
        Returns 'true' or 'false'.
        
        Parameters:
        -----------
        path : str
            Absolute path on the remote server.
        """
        result = history_manager.remote_file_exists(task_id=data.task_id, path=path)
        if result["ok"]:
            return "true" if result.get("exists") else "false"
        return json.dumps(result, ensure_ascii=False)

    def remote_mkdir(path: str) -> str:
        """
        Create a directory (and parents) on the remote server.
        
        Parameters:
        -----------
        path : str
            Absolute path on the remote server.
        """
        result = history_manager.remote_mkdir(task_id=data.task_id, path=path)
        return json.dumps(result, ensure_ascii=False)

    # === Summary Navigation Tools ===

    # DOC-BEGIN id=llm_handlers/list_summaries#1 type=tool v=1
    # summary: LLM 工具——列出指定目录下的文件/子目录及其 summary 概览
    # intent: LLM 通过此工具从项目根目录开始逐层下钻，了解项目结构。
    #   如果 path 为空则列出工作区根目录。返回格式包含每个条目的类型、summary、
    #   以及是否已展开（帮助 LLM 决定下一步 open 哪个节点）。
    def list_summaries(path: str = "") -> str:
        """
        List files and subdirectories at the given path with their summaries.
        Use this to explore the project structure layer by layer.
        Pass empty string or "/" for the workspace root.
        
        Parameters:
        -----------
        path : str
            Relative path from workspace root. Empty string for root.
        
        Returns: JSON with entries [{name, type, summary, is_open}]
        """
        rm_index = history_manager.get_rm_index(data.task_id)
        summary_state = history_manager.get_summary_state(data.task_id)
        if not rm_index or not summary_state:
            return json.dumps({"ok": False, "error": "Workspace not bound or still binding"})

        path = path.strip("/")
        prefix = path + "/" if path else ""

        entries = []
        for entry_key, entry_val in rm_index.dir_summary.items():
            if not entry_key.startswith(prefix):
                continue
            remainder = entry_key[len(prefix):]
            if not remainder:
                continue
            depth = remainder.rstrip("/").count("/")
            if depth > 0:
                continue

            node_id = f"{'dir' if entry_val['type'] == 'dir' else 'file'}::{entry_key}"
            entries.append({
                "name": remainder,
                "path": entry_key,
                "type": entry_val["type"],
                "summary": entry_val.get("summary", ""),
                "is_open": summary_state.is_open(node_id),
            })

        entries.sort(key=lambda e: (0 if e["type"] == "dir" else 1, e["name"]))
        return json.dumps({"ok": True, "path": path or "/", "entries": entries}, ensure_ascii=False)
    # DOC-END id=llm_handlers/list_summaries#1

    # DOC-BEGIN id=llm_handlers/open_summary#1 type=tool v=2
    # summary: LLM 工具——展开指定节点，仅返回操作是否成功（不返回内容，内容通过 pre_step_hook 注入下一步上下文）
    # intent: 支持三种节点类型：
    #   - dir::path/ → 展开显示目录下的直接子条目
    #   - file::path → 从远程加载文件的 DOC 块列表（触发懒加载）
    #   - doc::block_id → 标记 DOC 块为展开状态并刷新所属文件的折叠渲染
    #   展开后内容进入状态机，受 max_chars 限制和 LRU 淘汰管理。
    #   返回值仅包含 ok、node_id、chars_used、evicted 等状态字段，
    #   实际内容由 build_workspace_summary 在下一个 step 的 pre_step_hook 中注入上下文，
    #   避免 tool call 返回值与上下文注入重复消耗 token。
    def open_summary(node_id: str) -> str:
        """
        Expand a node to see its detailed content.
        
        Node ID formats:
        - "dir::src/services/" - expand a directory
        - "file::src/services/session_history.py" - expand a file (loads DOC blocks)
        - "doc::session_history/bind_workspace#2" - expand a specific DOC block
        
        Parameters:
        -----------
        node_id : str
            The node identifier to expand.
        """
        rm_index = history_manager.get_rm_index(data.task_id)
        summary_state = history_manager.get_summary_state(data.task_id)
        if not rm_index or not summary_state:
            return json.dumps({"ok": False, "error": "Workspace not bound or still binding"})

        parts = node_id.split("::", 1)
        if len(parts) != 2:
            return json.dumps({"ok": False, "error": f"Invalid node_id format: {node_id}. Use type::path"})

        node_type, node_path = parts

        if node_type == "dir":
            prefix = node_path if node_path.endswith("/") else node_path + "/"
            children = []
            for entry_key, entry_val in rm_index.dir_summary.items():
                if not entry_key.startswith(prefix):
                    continue
                remainder = entry_key[len(prefix):]
                if not remainder:
                    continue
                depth = remainder.rstrip("/").count("/")
                if depth > 0:
                    continue
                children.append(f"  {'📁' if entry_val['type'] == 'dir' else '📄'} {remainder}")
            children.sort()
            content = f"Directory: {node_path}\n" + "\n".join(children) if children else f"Directory: {node_path}\n  (empty)"
            result = summary_state.open_node(node_id, content)
            return json.dumps({"ok": True, "node_id": node_id, "chars_used": result.get("chars_used", 0), "evicted": result.get("evicted", [])}, ensure_ascii=False)

        # DOC-BEGIN id=llm_handlers/open_summary/file_branch#1 type=core v=2
        # summary: file 类型展开——读取文件完整代码，用 render_file_with_folding 折叠关闭的 DOC 块，
        #   结果作为该节点的 content 存入 summary_state，仅返回操作状态
        # intent: 核心设计：文件展开 ≠ 只列 DOC 块目录，而是显示完整代码。
        #   关闭的 DOC 块被折叠为一行占位符（包含 summary + 如何展开的提示），
        #   已展开的 DOC 块显示原始代码（递归处理嵌套子块）。
        #   content 存入 summary_state 后由 build_workspace_summary 在下一步上下文中呈现，
        #   tool call 返回值不再携带 content，避免与上下文注入重复。
        #   每次 open 都重新读取文件内容并重新渲染，确保内容最新。
        elif node_type == "file":
            blocks = rm_index.load_file_docs(node_path)
            file_content = rm_index.read_file_content(node_path)
            if file_content is None:
                return json.dumps({"ok": False, "error": f"Failed to read file: {node_path}"})

            if blocks:
                folded_content = summary_state.render_file_with_folding(file_content, blocks, node_path)
            else:
                folded_content = file_content

            content = f"File: {node_path}\n```\n{folded_content}\n```"
            result = summary_state.open_node(node_id, content)
            return json.dumps({"ok": True, "node_id": node_id, "chars_used": result.get("chars_used", 0), "evicted": result.get("evicted", [])}, ensure_ascii=False)
        # DOC-END id=llm_handlers/open_summary/file_branch#1

        # DOC-BEGIN id=llm_handlers/open_summary/doc_branch#1 type=core v=1
        # summary: doc 类型展开——标记该 DOC 块为 open 状态，然后刷新所属文件的折叠渲染
        # intent: 展开一个 DOC 块意味着该块对应的代码行从占位符恢复为原始代码。
        #   但展开后该块内部可能还有子 DOC 块（仍处于关闭状态），需要递归折叠。
        #   实现方式：先 open_node(doc::xxx) 标记状态，然后找到所属文件的 file::xxx 节点，
        #   如果文件节点也是展开的，则重新渲染文件内容并更新 file 节点的 content。
        #   返回值中包含更新后的文件折叠代码，让 LLM 立即看到变化。
        # DOC-BEGIN id=llm_handlers/open_summary/doc_branch#2 type=core v=3
        # summary: doc 类型展开——用轻量标记存入 _opened 表示展开状态，刷新所属文件折叠渲染，仅返回操作状态
        # intent: doc 节点不存储完整代码内容——其代码通过 file 节点的折叠渲染可见。
        #   _opened 中仅存一行标记（chars 极小），避免与 file 节点的折叠代码重复计费。
        #   展开 doc 块后必须刷新所属文件的折叠渲染：该块的代码行从占位符恢复为原始代码，
        #   但其内部子 DOC 块（若仍关闭）会递归折叠。
        #   如果所属文件节点未展开，返回 hint 引导 LLM 先展开文件。
        #   返回值不含 content，实际内容通过 pre_step_hook 注入下一步上下文。
        elif node_type == "doc":
            all_blocks = rm_index.get_all_loaded_blocks()
            block = None
            block_file = None
            for gkey, gval in all_blocks.items():
                if gval["id"] == node_path:
                    block = gval
                    block_file = gval.get("_file", "")
                    break
            if not block:
                return json.dumps({"ok": False, "error": f"DOC block not found: {node_path}. Open the file first."})

            # 轻量标记——仅记录展开状态，不存大段内容
            marker = f"[expanded] doc::{node_path} in {block_file}"
            result = summary_state.open_node(node_id, marker)

            # 刷新所属文件的折叠渲染（如果文件节点已展开）
            file_node_id = f"file::{block_file}"
            if summary_state.is_open(file_node_id):
                file_content = rm_index.read_file_content(block_file)
                file_blocks = rm_index.file_docs.get(block_file, {})
                if file_content is not None and file_blocks:
                    folded_content = summary_state.render_file_with_folding(file_content, file_blocks, block_file)
                    refreshed_content = f"File: {block_file}\n```\n{folded_content}\n```"
                    summary_state._update_node_content(file_node_id, refreshed_content)
                return json.dumps({"ok": True, "node_id": node_id, "chars_used": result.get("chars_used", 0), "evicted": result.get("evicted", [])}, ensure_ascii=False)
            else:
                return json.dumps({"ok": True, "node_id": node_id, "hint": f"DOC block expanded, but file '{block_file}' is not open. Use open_summary(\"file::{block_file}\") to see the code.", "chars_used": result.get("chars_used", 0), "evicted": result.get("evicted", [])}, ensure_ascii=False)
        # DOC-END id=llm_handlers/open_summary/doc_branch#2

        else:
            return json.dumps({"ok": False, "error": f"Unknown node type: {node_type}"})
    # DOC-END id=llm_handlers/open_summary#1

    # DOC-BEGIN id=llm_handlers/close_summary#1 type=tool v=1
    # summary: LLM 工具——关闭指定节点，释放上下文字符配额；若关闭的是 doc 块，同时刷新所属文件的折叠渲染
    # intent: LLM 在完成对某段代码的分析后，应主动关闭不再需要的节点来腾出空间。
    #   关闭 doc 块时，文件的折叠渲染需要更新（该块的代码行恢复为占位符），
    #   与 open_summary/doc_branch 保持对称。
    def close_summary(node_id: str) -> str:
        """
        Collapse a node to free up context space.
        
        Parameters:
        -----------
        node_id : str
            The node identifier to collapse (same format as open_summary).
        """
        summary_state = history_manager.get_summary_state(data.task_id)
        if not summary_state:
            return json.dumps({"ok": False, "error": "Workspace not bound"})
        result = summary_state.close_node(node_id)

        # DOC-BEGIN id=llm_handlers/close_summary/refresh_file#1 type=behavior v=1
        # summary: 如果关闭的是 doc 块，查找其所属文件并刷新折叠渲染
        # intent: 关闭 doc 块后，对应代码行需要重新变为占位符。
        #   通过 block_id 在 rm_index 的已加载 blocks 中查找所属文件，
        #   然后重新渲染并更新 file 节点的 content。
        parts = node_id.split("::", 1)
        if len(parts) == 2 and parts[0] == "doc" and rm_index:
            block_id = parts[1]
            all_blocks = rm_index.get_all_loaded_blocks()
            block_file = None
            for gkey, gval in all_blocks.items():
                if gval["id"] == block_id:
                    block_file = gval.get("_file", "")
                    break
            if block_file:
                file_node_id = f"file::{block_file}"
                if summary_state.is_open(file_node_id):
                    file_content = rm_index.read_file_content(block_file)
                    file_blocks = rm_index.file_docs.get(block_file, {})
                    if file_content is not None and file_blocks:
                        folded_content = summary_state.render_file_with_folding(
                            file_content, file_blocks, block_file
                        )
                        refreshed = f"File: {block_file}\n```\n{folded_content}\n```"
                        summary_state._update_node_content(file_node_id, refreshed)
        # DOC-END id=llm_handlers/close_summary/refresh_file#1

        return json.dumps({"ok": True, **result}, ensure_ascii=False)
    # DOC-END id=llm_handlers/close_summary#1

    # DOC-BEGIN id=llm_handlers/search_summaries#1 type=tool v=1
    # summary: LLM 工具——在所有已扫描的 summary 和 DOC 块中搜索关键词
    # intent: 当 LLM 需要找到特定功能或概念在代码库中的位置时，通过关键词搜索
    #   快速定位相关文件和 DOC 块，避免逐层下钻浪费 tool call 轮次。
    #   搜索范围包括：文件路径、dir_summary 的 summary/intent、已加载的 DOC 块的 summary/intent/id。
    def search_summaries(query: str) -> str:
        """
        Search for keywords across all file/directory summaries and loaded DOC blocks.
        Returns matching entries with their node IDs (can be passed to open_summary).
        
        Parameters:
        -----------
        query : str
            Search keyword (case-insensitive substring match).
        """
        rm_index = history_manager.get_rm_index(data.task_id)
        if not rm_index:
            return json.dumps({"ok": False, "error": "Workspace not bound"})

        query_lower = query.lower()
        results = []

        for entry_key, entry_val in rm_index.dir_summary.items():
            searchable = f"{entry_key} {entry_val.get('summary', '')} {entry_val.get('intent', '')}".lower()
            if query_lower in searchable:
                node_type = "dir" if entry_val["type"] == "dir" else "file"
                results.append({
                    "node_id": f"{node_type}::{entry_key}",
                    "type": entry_val["type"],
                    "path": entry_key,
                    "summary": entry_val.get("summary", ""),
                    "match": "path/summary",
                })

        all_blocks = rm_index.get_all_loaded_blocks()
        for gkey, block in all_blocks.items():
            searchable = f"{block['id']} {block.get('summary', '')} {block.get('intent', '')}".lower()
            if query_lower in searchable:
                results.append({
                    "node_id": f"doc::{block['id']}",
                    "type": "doc_block",
                    "file": block.get("_file", ""),
                    "id": block["id"],
                    "summary": block.get("summary", ""),
                    "match": "doc_block",
                })

        return json.dumps({"ok": True, "query": query, "count": len(results), "results": results[:30]}, ensure_ascii=False)
    # DOC-END id=llm_handlers/search_summaries#1

    task_dir=f"{DATA_DIR}/tasks/{data.task_id}"
    tc = Tool_Calls(LOG_DIR=task_dir, MAX_CHAR=800000, mode="Summary")
    tc.remove_temporary_context()

    # DOC-BEGIN id=llm_handlers/workspace_interceptor_setup#2 type=behavior v=2
    # summary: 当 workspace 已 bound 时，构造 output_interceptor 闭包、hash_to_path 映射、
    #   以及 build_workspace_summary 函数（每 step 动态生成最新 summary 文本）
    # intent: workspace bound 模式下 summary 不再一次性注入 tc，而是作为临时消息在每个 step
    #   开始前注入、step 结束后删除。这是因为每个 step 都可能改变文件内容（§ 协议写入）
    #   或改变 summary 展开状态（open_summary/close_summary 工具调用），所以 summary 必须
    #   在每个 step 前重新生成。build_workspace_summary() 封装了生成逻辑，
    #   返回最新的 workspace context 文本，供 pre_step_hook 注入 tc。
    workspace_interceptor = None
    ws_info = history_manager.get_workspace_info(task_id=data.task_id)
    is_workspace_bound = ws_info is not None and ws_info.get("status") == "bound"
    build_workspace_summary = None
    rm_index = None
    hash_to_path_map = {}

    if is_workspace_bound:
        from src.services.workspace_edit_parser import make_interceptor

        rm_index = history_manager.get_rm_index(data.task_id)
        if rm_index:
            work_dir = rm_index.work_dir
            for entry_key, entry_val in rm_index.dir_summary.items():
                if entry_val["type"] == "file":
                    abs_path = f"{work_dir}/{entry_key}"
                    h = str(abs(hash(abs_path)) % (10**10))
                    hash_to_path_map[h] = abs_path

        # DOC-BEGIN id=llm_handlers/ws_file_writer#2 type=behavior v=2
        # summary: 远程文件写入封装——支持绝对路径和相对路径，写入后自动失效 RMIndex 缓存、
        #   更新 hash 映射、更新 dir_summary（为新文件添加条目）
        # intent: § Start/End/Replace 使用绝对路径（通过 hash_to_path 查出），
        #   § Touch/Write 使用相对路径（LLM 直接写路径）。需要统一处理两种情况。
        #   写入成功后要：(1) invalidate RMIndex 缓存使下次读取重新扫描 DOC 块，
        #   (2) 为新文件添加 hash_to_path 映射和 dir_summary 条目，
        #   使后续 § Start/End/Replace 和 summary 刷新都能看到新文件。
        def _ws_file_writer(path: str, content: str) -> bool:
            if not path.startswith("/") and rm_index:
                abs_path = f"{rm_index.work_dir}/{path}"
            else:
                abs_path = path
            parent_dir = "/".join(abs_path.rsplit("/", 1)[:-1])
            if parent_dir:
                history_manager.remote_mkdir(data.task_id, parent_dir)
            result = history_manager.remote_write_file(data.task_id, abs_path, content)
            if result.get("ok"):
                if rm_index:
                    rel = abs_path.replace(rm_index.work_dir + "/", "")
                    rm_index.invalidate_file(rel)
                    h = str(abs(hash(abs_path)) % (10**10))
                    if h not in hash_to_path_map:
                        hash_to_path_map[h] = abs_path
                    if rel not in rm_index.dir_summary:
                        rm_index.dir_summary[rel] = {"type": "file", "summary": "", "intent": ""}
                return True
            return False
        # DOC-END id=llm_handlers/ws_file_writer#2

        # DOC-BEGIN id=llm_handlers/ws_file_reader#1 type=behavior v=1
        # summary: 远程文件读取封装——同样支持相对路径和绝对路径
        # intent: 与 _ws_file_writer 保持一致的路径处理逻辑
        def _ws_file_reader(path: str):
            if not path.startswith("/") and rm_index:
                abs_path = f"{rm_index.work_dir}/{path}"
            else:
                abs_path = path
            result = history_manager.remote_read_file(data.task_id, abs_path)
            if result.get("ok"):
                return result["content"]
            return None
        # DOC-END id=llm_handlers/ws_file_reader#1

        # DOC-BEGIN id=llm_handlers/ws_binary_reader#1 type=behavior v=1
        # summary: 远程二进制文件读取封装——支持相对路径和绝对路径，调用 remote_read_file_binary
        # intent: § Exec push 指令需要读取图片等二进制文件。与 _ws_file_reader 保持一致的路径处理逻辑。
        #   返回 dict 格式与 remote_read_file_binary 一致：{"ok": True, "data": bytes, "size": int}
        def _ws_binary_reader(path: str) -> dict:
            if not path.startswith("/") and rm_index:
                abs_path = f"{rm_index.work_dir}/{path}"
            else:
                abs_path = path
            return history_manager.remote_read_file_binary(task_id=data.task_id, path=abs_path)
        # DOC-END id=llm_handlers/ws_binary_reader#1

        workspace_interceptor = make_interceptor(
            hash_to_path_map, _ws_file_reader, _ws_file_writer,
            ec=ec, binary_reader=_ws_binary_reader,
        )

        # DOC-BEGIN id=llm_handlers/build_workspace_summary#2 type=core v=2
        # summary: 构造 build_workspace_summary 闭包——每次调用时从 rm_index、hash_to_path_map、
        #   summary_state、data.pinned_codes 的最新状态生成 workspace context 文本
        # intent: 这个函数会在每个 LLM step 开始前被 pre_step_hook 调用。因为 rm_index.dir_summary、
        #   hash_to_path_map、summary_state 都是可变对象/闭包变量，每次调用都能拿到最新数据。
        #   生成的文本包含五部分：(1) 目录结构概览 (2) hash 映射表 (3) 已展开节点内容
        #   (4) DOC 块摘要 (5) 用户 pinned 的代码内容。
        #   data.pinned_codes 也通过闭包引用，workspace 模式下不走 inject_temporary_context，
        #   而是统一在此处渲染，确保 pinned code 和 workspace summary 作为同一条临时消息注入/删除。
        def build_workspace_summary() -> str:
            summary_state = history_manager.get_summary_state(data.task_id)
            lines = ["[WORKSPACE_SUMMARY] Bound workspace — use Hash for § Start/End/Replace, use relative path for § Write/Touch\n"]

            lines.append("== Project Structure ==")
            if rm_index:
                sorted_entries = sorted(
                    rm_index.dir_summary.items(),
                    key=lambda e: (e[0].count("/"), 0 if e[1]["type"] == "dir" else 1, e[0])
                )
                for entry_key, entry_val in sorted_entries:
                    depth = entry_key.rstrip("/").count("/")
                    indent = "  " * depth
                    icon = "📁" if entry_val["type"] == "dir" else "📄"
                    summary_hint = entry_val.get("summary", "")
                    if entry_val["type"] == "file":
                        abs_path = f"{rm_index.work_dir}/{entry_key}"
                        h = str(abs(hash(abs_path)) % (10**10))
                        lines.append(f"{indent}{icon} {entry_key} (Hash: {h})")
                    else:
                        lines.append(f"{indent}{icon} {entry_key}")
                    if summary_hint:
                        lines.append(f"{indent}   └─ {summary_hint[:100]}")
            lines.append("")

            lines.append("== File Hash Table (for § Start/End/Replace) ==")
            for h, p in hash_to_path_map.items():
                rel = p.replace(rm_index.work_dir + "/", "") if rm_index else p
                lines.append(f"  {h} → {rel}")
            lines.append("")

            # DOC-BEGIN id=llm_handlers/build_workspace_summary/render_opened_nodes#1 type=core v=1
            # summary: 遍历所有已展开节点，对 file:: 节点重新执行折叠渲染后输出，其余节点直接输出 cached content
            # intent: file:: 节点的折叠渲染依赖其内部 doc 块的展开/关闭状态。由于 doc 块可能被
            #   LRU 自动淘汰（_auto_evict_locked 不会触发文件刷新），file 节点的 cached content
            #   可能与实际的 doc 展开状态不一致。每次 build_workspace_summary 时对 file 节点
            #   重新渲染，确保 LLM 看到的折叠代码始终反映最新状态。
            #   doc:: 节点不单独输出内容——其代码已经内嵌在 file 节点的折叠渲染中，
            #   单独输出 doc_info 是冗余的。doc:: 节点在 _opened 中仅作为展开状态标记存在。
            if summary_state:
                opened_info = summary_state.get_opened_list()
                if opened_info["opened_count"] > 0:
                    lines.append(f"== Expanded Summary Nodes ({opened_info['total_chars']}/{opened_info['max_chars']} chars) ==")
                    for node_info in opened_info["nodes"]:
                        nid = node_info["node_id"]
                        if not summary_state.is_open(nid):
                            continue
                        parts_nid = nid.split("::", 1)
                        if len(parts_nid) == 2 and parts_nid[0] == "file" and rm_index:
                            # 对 file 节点重新折叠渲染，确保反映最新 doc 展开状态
                            rel_path = parts_nid[1]
                            file_blocks = rm_index.file_docs.get(rel_path, {})
                            file_content = rm_index.read_file_content(rel_path)
                            if file_content is not None and file_blocks:
                                folded = summary_state.render_file_with_folding(file_content, file_blocks, rel_path)
                                refreshed = f"File: {rel_path}\n```\n{folded}\n```"
                                summary_state._update_node_content(nid, refreshed)
                                lines.append(f"\n--- {nid} ---")
                                lines.append(refreshed)
                            elif file_content is not None:
                                refreshed = f"File: {rel_path}\n```\n{file_content}\n```"
                                summary_state._update_node_content(nid, refreshed)
                                lines.append(f"\n--- {nid} ---")
                                lines.append(refreshed)
                            else:
                                entry = summary_state._opened.get(nid)
                                if entry:
                                    lines.append(f"\n--- {nid} ---")
                                    lines.append(entry["content"])
                        elif len(parts_nid) == 2 and parts_nid[0] == "doc":
                            # doc 节点不单独输出——其代码已在所属 file 节点的折叠渲染中可见
                            continue
                        else:
                            # dir 节点或其他类型，直接输出 cached content
                            entry = summary_state._opened.get(nid)
                            if entry:
                                lines.append(f"\n--- {nid} ---")
                                lines.append(entry["content"])
                    lines.append("")
            # DOC-END id=llm_handlers/build_workspace_summary/render_opened_nodes#1

            # DOC-BEGIN id=llm_handlers/build_workspace_summary/doc_blocks_for_closed_files#1 type=behavior v=1
            # summary: 仅对未展开的文件显示 DOC 块摘要列表，已展开文件的 DOC 块已内嵌在折叠代码中
            # intent: 文件已展开时，DOC 块信息通过 render_file_with_folding 内嵌在代码中（以折叠占位符或原始代码形式），
            #   再在此处重复列出会浪费 token。仅对已加载但未展开的文件显示 DOC 块目录，
            #   让 LLM 知道这些文件有哪些 DOC 块可以探索。
            if rm_index:
                has_doc_header = False
                for rel_path, blocks in rm_index.file_docs.items():
                    if not blocks:
                        continue
                    file_node_id = f"file::{rel_path}"
                    if summary_state and summary_state.is_open(file_node_id):
                        continue  # 文件已展开，DOC 块已在折叠代码中可见
                    if not has_doc_header:
                        lines.append("== DOC Blocks in Closed Files (open file to see code) ==")
                        has_doc_header = True
                    lines.append(f"\n  File: {rel_path}")
                    for bid, block in blocks.items():
                        lines.append(f"    [DOC] {bid}: {block['summary'][:80]}  {{...}}")
                if has_doc_header:
                    lines.append("")
            # DOC-END id=llm_handlers/build_workspace_summary/doc_blocks_for_closed_files#1

            # DOC-BEGIN id=llm_handlers/build_workspace_summary/pinned_codes#1 type=behavior v=1
            # summary: 将用户 pinned 的代码块渲染进 workspace summary，与 TEMPORARY_CONTEXT 等价
            # intent: workspace bound 模式下不走 inject_temporary_context（因为 summary 每 step 重建），
            #   pinned_codes 统一在此处渲染，确保 pinned code 生命周期与 workspace summary 一致。
            #   格式与原有 inject_temporary_context 保持兼容（File + Hash + 代码块）。
            if data.pinned_codes:
                lines.append("== Pinned Code Files ==")
                for p in data.pinned_codes:
                    lines.append(f"\nFile: {p.filename} (Hash: {p.hash})")
                    lines.append(f"```\n{p.content}\n```")
                lines.append("")
            # DOC-END id=llm_handlers/build_workspace_summary/pinned_codes#1

            lines.append("[END WORKSPACE_SUMMARY]")
            return "\n".join(lines)
        # DOC-END id=llm_handlers/build_workspace_summary#2
    # DOC-END id=llm_handlers/workspace_interceptor_setup#2

    # DOC-BEGIN id=llm_handlers/trim_history_on_demand#1 type=behavior v=1
    # summary: 当 data.trim_history 为 True 时，对历史记录执行两步削减：
    #   (1) 将所有 § Start/End/Replace 编辑协议块替换为占位符
    #   (2) 将被 pinned_codes 完全覆盖的代码块替换为占位符
    # intent: 用户主动触发时才削减，避免每次请求都执行正则扫描的性能开销。
    #   削减必须在 remove_temporary_context 之后、inject_temporary_context 之前执行，
    #   因为旧的 TEMPORARY_CONTEXT 已被移除，而新的尚未注入，此时扫描不会误伤当前 pinned 内容。
    if data.trim_history:
        tc.trim_edit_protocol_blocks()
        if data.pinned_codes:
            tc.trim_redundant_code_blocks(data.pinned_codes)
    # DOC-END id=llm_handlers/trim_history_on_demand#1

    # DOC-BEGIN id=llm_handlers/context_injection_branch#3 type=behavior v=3
    # summary: 根据是否 workspace bound 分支注入上下文：bound 时 pinned_codes 合并进 workspace summary
    #   （由 pre_step_hook 动态注入），未 bound 时使用原有 inject_temporary_context 注入
    # intent: workspace bound 模式下 summary 是临时的——每个 LLM step 前注入、step 后删除。
    #   但用户仍然可能 pin 了代码（比如非工作区内的片段、或本地笔记），
    #   这些 pinned_codes 需要包含在 workspace summary 中一起注入，不能丢弃。
    #   非 workspace 场景则走原有的 inject_temporary_context 逻辑。
    if not is_workspace_bound:
        if data.pinned_codes:
            tc.inject_temporary_context(data.pinned_codes)
    # DOC-END id=llm_handlers/context_injection_branch#3

    # DOC-BEGIN id=llm_handlers/step_hooks#1 type=core v=1
    # summary: 构造 pre_step_hook 和 post_step_hook 回调，在 query_with_tools 的每个 step 前后
    #   动态注入/删除 workspace summary 临时消息
    # intent: pre_step_hook 在每个 step 的 LLM 调用之前执行：先删除上一步的旧 summary，
    #   然后生成最新 summary 并注入 tc。post_step_hook 在对话彻底结束后清理最后一条 summary。
    #   非 workspace 模式下两个 hook 都为 None，query_with_tools 会跳过调用。
    #   使用 tc 的 inject_workspace_summary / remove_workspace_summary 方法操作，
    #   这两个方法通过 [WORKSPACE_SUMMARY] 标记识别临时消息。
    pre_step_hook = None
    post_step_hook = None
    if is_workspace_bound and build_workspace_summary is not None:
        def pre_step_hook():
            tc.remove_workspace_summary()
            summary_text = build_workspace_summary()
            tc.inject_workspace_summary(summary_text)

        # DOC-BEGIN id=llm_handlers/post_step_hook/save_summary#1 type=behavior v=1
        # summary: post_step_hook 在对话结束后清理 workspace summary，并同步保存 summary 展开状态
        # intent: _schedule_save 是异步延迟 1 秒的 debounce，对话结束时 Timer 可能尚未触发。
        #   save_now() 取消 pending Timer 并同步写入，确保本轮对话中的 open/close 操作
        #   在对话结束前持久化到远程 .rm/summary_state.json。
        #   如果 summary_state 不存在（非 workspace 场景），静默跳过。
        def post_step_hook():
            tc.remove_workspace_summary()
            summary_state = history_manager.get_summary_state(data.task_id)
            if summary_state:
                summary_state.save_now()
        # DOC-END id=llm_handlers/post_step_hook/save_summary#1
    # DOC-END id=llm_handlers/step_hooks#1

    # DOC-BEGIN id=llm_handlers/inject_timestamp#1 type=behavior v=1
    # summary: 在 data.question（user prompt）前注入当前服务器时间戳（英文格式，精确到秒）
    # intent: 让 LLM 感知当前时间，便于生成带时间上下文的回答（如日志分析、定时任务等场景）。
    #   时间戳放在 prompt 最前面用方括号包裹，与用户原始问题用换行分隔，不影响原始语义。
    #   使用 datetime.now() 取服务器本地时间；若需要 UTC 可后续切换为 datetime.utcnow()。
    from datetime import datetime as _dt
    _now_str = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    data.question = f"[Current Time: {_now_str}]\n{data.question}"
    # DOC-END id=llm_handlers/inject_timestamp#1

    active_sessions = history_manager.list_task_sessions(data.task_id)
    has_visible_terminal = any(
        not history_manager.is_terminal_bound(sid) 
        for sid in active_sessions
    )
    tools = []
    ct = custom_tools()
    if data.enable_fc:
        tools.extend([
            ct.func_search,
        ])
        
        if has_visible_terminal:
            tools.extend([
                # terminal_create_new, 
                terminal_send_command, 
                terminal_exec_command,
                # terminal_get_status, 
                # terminal_terminate_session,
                # terminal_stage_command,
            ])
        
        if is_workspace_bound:
            tools.extend([
                open_summary,
            ])
            
    
    def system_prompt() -> str:
        # DOC-BEGIN id=llm_handlers/system_prompt/terminal_context#2 type=behavior v=2
        # summary: 拉取当前 task 可见的 terminal 列表，排除所有 is_bound=true 的终端，将剩余终端屏幕快照注入上下文
        # intent: is_bound=true 的 terminal 专用于远程文件操作（marker 协议），其屏幕内容对 LLM 无帮助。
        #   通过 is_terminal_bound() 判断固有属性，而非旧的 workspace binding 查找。
        # DOC-BEGIN id=llm_handlers/system_prompt/terminal_llm_snapshot#1 type=behavior v=2
        # summary: 拉取当前 task 可见的 terminal 列表，对每个终端调用 get_llm_status() 获取裁剪后的屏幕快照注入上下文
        # intent: PTY 屏幕是 1000×1000 的大缓冲区，全量注入 LLM 上下文会浪费大量 token。
        #   get_llm_status() 裁剪宽度到 120 字符、仅保留尾部 80 行、总字符数限制 8000（约 2000 token），
        #   头部 banner/motd 对 LLM 决策无帮助，只保留最近输出即可。
        #   通过 manager.get_session(sid) 直接获取底层 TerminalSession 对象，
        #   如果获取失败则 fallback 到 history_manager.get_session_status() 保证兼容性。
        from src.services.terminal import manager as terminal_manager
        active_sessions = history_manager.list_task_sessions(data.task_id)

        terminal_context_block = "\n=== ACTIVE TERMINAL SESSIONS ===\n"
        visible_count = 0
        for sid in active_sessions:
            if history_manager.is_terminal_bound(sid):
                continue
            session_obj = terminal_manager.get_session(sid)
            if session_obj and hasattr(session_obj, 'get_llm_status'):
                status = session_obj.get_llm_status()
            else:
                status = history_manager.get_session_status(data.task_id, sid)
            terminal_context_block += f"--- Session ID: {sid} ---\n{status}\n"
            visible_count += 1
        # DOC-END id=llm_handlers/system_prompt/terminal_llm_snapshot#1
        if visible_count == 0:
            terminal_context_block += "No active terminal sessions in this task.\n"
        terminal_context_block += "================================\n"
        # DOC-END id=llm_handlers/system_prompt/terminal_context#2

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
                "# summary: <必须包含核心输入、核心逻辑、输出 / 返回 / 核心效果（函数需说明入参类型 / 含义、核心计算 / 处理逻辑、返回值类型 / 含义；非函数代码段需说明操作对象、核心操作、操作结果）>\n"
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
                "# summary: 函数add接收两个可为None的数值型参数a和b，将None视为0后计算两者的和，返回求和后的数值结果\n"
                "# intent: 演示“注释块必须包裹代码段”的位置规则；同时说明当存在边界条件策略（如 None 处理）时，应对局部决策单独加注释，避免把关键约定埋在实现细节里\n"
                "def add(a, b):\n"
                "    # DOC-BEGIN id=example/math/add/none-as-zero#1 type=behavior v=1\n"
                "    # summary: 检查参数a是否为None，若为None则将其赋值为0，完成输入参数的归一化\n"
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
                "6. My project code is NOT available online or in any public repository. "
                "If you need to see any source code (e.g. src/services/xxx.py), "
                "do NOT guess or fabricate it — ask me directly and I will pin it for you."
            )
        ]
        
        non_workspace_hint = ""
        if not is_workspace_bound:
            non_workspace_hint = (
                "\n[IMPORTANT] Do NOT use the terminal to search for or read project source code. "
                "The pinned code files here are unrelated to any terminal session. "
                "If you need to see any project code (e.g. src/services/xxx.py, tests/xxx.py), "
                "ask me directly and I will pin it for you.\n"
            )
        else:
            # DOC-BEGIN id=llm_handlers/system_prompt/workspace_hint_with_exec#1 type=behavior v=1
            # summary: workspace bound 时的提示文本——包含 summary 工具用法 + § Exec push 图片推送协议
            # intent: § Exec 协议告知 LLM 如何将代码生成的图片发送给用户。
            #   每次 § Exec 块内只允许一条命令，避免 LLM 批量推送导致传输阻塞。
            #   push 的 remote_path 支持相对路径（相对工作目录）和绝对路径。
            #   display_name 应包含扩展名，后端据此推断 MIME 类型。
            non_workspace_hint = (
                "\n[IMPORTANT] Workspace is bound. To view project source code, use the summary navigation tools:\n"
                "- open_summary(\"file::path/to/file.py\") to view a file with folded DOC blocks\n"
                "- open_summary(\"dir::path/to/dir/\") to expand a directory\n"
                "- open_summary(\"doc::block_id\") to expand a specific DOC block\n"
                "- search_summaries(\"keyword\") to search across all summaries and DOC blocks\n"
                "Do NOT use terminal commands (cat, less, head, etc.) to read project source code. "
                "The summary tools provide structured, token-efficient access to the codebase.\n"
                "\n=== IMAGE / FILE PUSH PROTOCOL ===\n"
                "To display an image (or any binary file) to the user, use the § Exec protocol.\n"
                "Format (ONE command per block):\n\n"
                "§ Exec\n"
                "```bash\n"
                "push <remote_path> as <display_name>\n"
                "```\n"
                "§ Exec\n\n"
                "Example — after generating a plot with matplotlib:\n\n"
                "§ Exec\n"
                "```bash\n"
                "push output/plot.png as scatter_plot.png\n"
                "```\n"
                "§ Exec\n\n"
                "Rules:\n"
                "- <remote_path>: relative to workspace root, or absolute path\n"
                "- <display_name>: filename with extension (used for MIME type detection)\n"
                "- Only ONE push command per § Exec block\n"
                "- The image will be sent inline in the chat for the user to see\n"
            )
            # DOC-END id=llm_handlers/system_prompt/workspace_hint_with_exec#1
        
        # DOC-BEGIN id=llm_handlers/system_prompt/code_edit_protocol#2 type=behavior v=2
        # summary: 根据 is_workspace_bound 和 system_prompt_mode 构造 code_edit_protocol 文本：
        #   workspace bound 时精简描述 § Start/End/Replace + § Write 两套协议，
        #   非 workspace 时仅包含 § Start/End/Replace 协议
        # intent: workspace bound 模式下工具（list_summaries 等）已通过 tools 参数自动暴露给 LLM，
        #   prompt 中无需重复描述工具用途。§ Write 示例使用代码块包裹格式（推荐但非强制）。
        #   精简 prompt 减少 token 消耗，同时保留 LLM 正确使用协议所需的最少信息。
        # DOC-BEGIN id=llm_handlers/system_prompt/code_edit_default#1 type=behavior v=1
        # summary: 非 workspace 模式下的 code_edit_protocol——仅包含 § Start/End/Replace 协议
        # intent: 非 workspace 模式下编辑协议由前端解析执行（前端对 pinned 文件做 diff apply），
        #   LLM 仍然需要知道 § 协议的语法才能正确输出 patch。
        #   workspace 模式下的分支会覆盖此默认值。
        code_edit_protocol = (
            "\n=== CODE EDIT PROTOCOL ===\n"
            "You can propose edits to PINNED code blocks (identified by Hash) using the following format. "
            "DO NOT output the full file; ONLY provide the patch using this EXACT structure:\n\n"
            "§ Start (hash_value)\n"
            "```[programming language identifier, e.g. python/javascript/bash]\n"
            "[precise lines of existing code where the edit begins, must be uniquely matched]\n"
            "```\n"
            "§ Start (hash_value)\n\n"
            "§ End (hash_value)\n"
            "```[programming language identifier, e.g. python/javascript/bash]\n"
            "[precise lines of existing code where the edit ends, must be uniquely matched]\n"
            "```\n"
            "§ End (hash_value)\n\n"
            "§ Replace (hash_value)\n"
            "```[programming language identifier, e.g. python/javascript/bash]\n"
            "[The new code that replaces everything between and including the Start/End anchors]\n"
            "```\n"
            "§ Replace (hash_value)\n\n"
            "RULES FOR EDITING:\n"
            "- The text inside 'Start' and 'End' MUST be copied verbatim from the pinned file and matched by strict character-for-character comparison (including whitespace and blank lines).\n"
            "- Each anchor snippet MUST occur exactly once in the pinned file (occurrence count == 1). If not, DO NOT produce a patch; ask for more surrounding lines to make anchors unique.\n"
            "- You will find the current content of pinned files in the [TEMPORARY_CONTEXT] assistant message provided in the history.\n"
            "- Refer to files ONLY by their Hash value.\n"
            "- The user may have only pinned a SUBSET of the relevant files. If you need to see or edit a file "
            "that is NOT currently pinned, tell the user which file you need and they will pin it for you.\n"
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
        # DOC-END id=llm_handlers/system_prompt/code_edit_default#1
        if is_workspace_bound:
            code_edit_protocol = (
                    "\n=== CODE EDIT PROTOCOL ===\n"
                    "You can propose edits to PINNED code blocks (identified by Hash) using the following format. "
                    "DO NOT output the full file; ONLY provide the patch using this EXACT structure:\n\n"
                    "§ Start (hash_value)\n"
                    "```[programming language identifier, e.g. python/javascript/bash]\n"
                    "[precise lines of existing code where the edit begins, must be uniquely matched]\n"
                    "```\n"
                    "§ Start (hash_value)\n\n"
                    "§ End (hash_value)\n"
                    "```[programming language identifier, e.g. python/javascript/bash]\n"
                    "[precise lines of existing code where the edit ends, must be uniquely matched]\n"
                    "```\n"
                    "§ End (hash_value)\n\n"
                    "§ Replace (hash_value)\n"
                    "```[programming language identifier, e.g. python/javascript/bash]\n"
                    "[The new code that replaces everything between and including the Start/End anchors]\n"
                    "```\n"
                    "§ Replace (hash_value)\n\n"
                    "RULES FOR EDITING:\n"
                    "- The text inside 'Start' and 'End' MUST be copied verbatim from the pinned file and matched by strict character-for-character comparison (including whitespace and blank lines).\n"
                    "- Each anchor snippet MUST occur exactly once in the pinned file (occurrence count == 1). If not, DO NOT produce a patch; ask for more surrounding lines to make anchors unique.\n"
                    "- You will find the current content of pinned files in the [TEMPORARY_CONTEXT] assistant message provided in the history.\n"
                    "- Refer to files ONLY by their Hash value.\n"
                    "- The user may have only pinned a SUBSET of the relevant files. If you need to see or edit a file "
                    "that is NOT currently pinned, tell the user which file you need and they will pin it for you.\n"
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
            
        return "\n".join(base_rules) + "\n" + terminal_context_block + "\n" + code_edit_protocol + "\n" + non_workspace_hint + "\n" + pending_context

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
            output_interceptor=workspace_interceptor,
            pre_step_hook=pre_step_hook,
            post_step_hook=post_step_hook,
        )
        # DOC-BEGIN id=llm_handlers/submit_fn_to_executor#2 type=behavior v=2
        # summary: 将 query_with_tools 偏函数包装为 _run_fn，提交到线程池执行；
        #   _run_fn 在 fn 正常或异常结束后都向 result_queue 发送 None（EOS 信号）
        # intent: query_with_tools 内部使用同步 requests 阻塞调用 LLM API，不能在 asyncio 事件循环中直接执行。
        #   原来直接 run_in_executor(None, fn) 有一个隐患：fn 执行结束后没有人向 result_queue
        #   发送 None，导致 event_generator 永远阻塞在 result_queue.get() 上，SSE 连接无法关闭。
        #   现在用 _run_fn 包装，在 finally 中保证 EOS 一定发送。这是唯一的 EOS 来源。
        def _run_fn():
            try:
                fn()
            except Exception:
                traceback.print_exc()
            finally:
                result_queue.put(None)
        loop.run_in_executor(None, _run_fn)
        # DOC-END id=llm_handlers/submit_fn_to_executor#2
    except Exception:
        result_queue.put(None)
        traceback.print_exc()
        
    # DOC-BEGIN id=llm_handlers/event_generator_with_image#1 type=core v=1
    # summary: SSE 事件生成器——从 result_queue 逐条取出消息，区分纯文本 (type=llm_info) 和
    #   结构化事件 (type=image)，分别以不同格式 yield 给前端
    # intent: 原有协议只输出纯文本（content 字符串直接 yield），无法传递图片等二进制数据。
    #   扩展后约定：type="image" 事件以 "§IMG§" + JSON 的格式 yield，
    #   前端通过检测此前缀区分文本和结构化事件。选择内联标记而非切换 media_type，
    #   是因为 SSE 流中文本和图片事件交替出现，不能中途改 Content-Type。
    #   "§IMG§" 前缀足够独特，不会与正常 LLM 输出冲突。
    async def event_generator():
        while True:
            item = await loop.run_in_executor(None, result_queue.get)
            if item is None:
                break
            if isinstance(item, dict):
                msg_type = item.get("type", "")
                if msg_type == "image":
                    yield "§IMG§" + json.dumps(item["data"], ensure_ascii=False)
                else:
                    c = item.get("data", {}).get("content", "")
                    if c:
                        yield c
    return StreamingResponse(event_generator(), media_type="text/plain; charset=utf-8")
    # DOC-END id=llm_handlers/event_generator_with_image#1
