# src/handlers/llm_handlers.py
import os
import asyncio
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
