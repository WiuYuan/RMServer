# src/services/session_history.py
import threading
from typing import Dict, List, Optional
from src.services.terminal import manager, TerminalSession
from src.config import RECORDINGS_ROOT
import time
import uuid
import json
import os
import logging

logger = logging.getLogger(__name__)

class TaskHistoryManager:
    """
    任务历史管理器（Terminal 全局化 + Workspace 一等公民架构）
    Terminal 是全局独立资源，有固有属性 is_bound（是否为文件操作专用）。
    Workspace 是独立实体，包含 1 个 bound terminal + 1 个 interactive terminal。
    Task 通过 mount 挂载 Terminal，通过 attach 关联 Workspace。
    """
    def __init__(self):
        self._lock = threading.Lock()

        # DOC-BEGIN id=session_history/global_terminals#2 type=state v=2
        # summary: 全局 terminal 池，session_id → {"created_at": float, "is_bound": bool}
        # intent: Terminal 是全局独立资源。is_bound 是固有属性：
        #   is_bound=True 的 terminal 专用于远程文件操作（marker 协议），屏幕快照不注入 LLM。
        #   is_bound=False 的 terminal 是普通交互终端。用户可以通过 toggle_terminal_bound 切换。
        self._global_terminals: Dict[str, dict] = {}
        # DOC-END id=session_history/global_terminals#2

        # DOC-BEGIN id=session_history/task_mounts#1 type=state v=1
        # summary: task 挂载关系，task_id → set(session_id)，记录每个 task 挂载了哪些 terminal
        # intent: 一个 terminal 可以被多个 task 同时挂载。
        #   挂载关系决定前端过滤、以及 LLM system prompt 中注入哪些终端的屏幕快照。
        self._task_mounts: Dict[str, set] = {}
        # DOC-END id=session_history/task_mounts#1

        self._pending_actions: Dict[str, Optional[dict]] = {}

        # DOC-BEGIN id=session_history/workspaces#1 type=state v=1
        # summary: 全局 workspace 池，workspace_id → {bound_terminal_id, interactive_terminal_id, status, work_dir, message, ...}
        # intent: Workspace 是一等公民独立实体，不属于任何 task。包含两个 terminal 引用：
        #   - bound_terminal_id: is_bound=True 的文件操作专用终端
        #   - interactive_terminal_id: is_bound=False 的用户交互终端
        #   status 字段跟踪生命周期：creating → binding → bound → error
        #   Task 通过 _task_workspace 关联 workspace。
        self._workspaces: Dict[str, dict] = {}
        # DOC-END id=session_history/workspaces#1

        # DOC-BEGIN id=session_history/task_workspace#1 type=state v=1
        # summary: task → workspace 关联，task_id → workspace_id，每个 task 最多关联一个 workspace
        # intent: 关联后 task 获得该 workspace 的文件操作能力（通过 bound terminal）和
        #   interactive terminal 的自动 mount。解除关联时清除 task 级缓存（rm_index、summary_state）。
        self._task_workspace: Dict[str, str] = {}
        # DOC-END id=session_history/task_workspace#1

        # DOC-BEGIN id=session_history/rm_indexes#2 type=state v=2
        # summary: 每个 workspace 的 RMIndex 实例，workspace_id → RMIndex
        # intent: RMIndex 绑定到 workspace（而非 task），因为同一个 workspace 可以被多个 task attach。
        #   通过 get_rm_index(task_id) 查找 task → workspace → RMIndex 链。
        self._rm_indexes: Dict[str, "RMIndex"] = {}
        # DOC-END id=session_history/rm_indexes#2

        self._summary_states: Dict[str, "SummaryStateManager"] = {}
        # DOC-END id=session_history/summary_states#2

        # DOC-BEGIN id=session_history/recordings#1 type=state v=1
        # summary: 录制存储——保存的 terminal 录制和 workspace 录制，以及正在进行的回放状态
        # intent: _active_replays 记录当前正在回放的 session_id → stop_event，用于停止回放。
        #   录制文件持久化到 DATA_DIR/recordings/ 目录下，JSON 格式。
        #   terminal 录制格式: {"type": "terminal", "name": str, "commands": [...], "created_at": float}
        #   workspace 录制格式: {"type": "workspace", "name": str,
        #     "bound_commands": [...], "interactive_commands": [...], "created_at": float}
        self._active_replays: Dict[str, threading.Event] = {}  # session_id → stop_event
        self._recordings_dir = RECORDINGS_ROOT
        os.makedirs(self._recordings_dir, exist_ok=True)
        # DOC-END id=session_history/recordings#1

    # === 全局 Terminal 管理 ===

    # DOC-BEGIN id=session_history/create_terminal#2 type=core v=2
    # summary: 创建一个全局 terminal（不属于任何 task），支持 is_bound 属性，返回 session_id
    # intent: is_bound 默认 False（普通交互终端）。创建时如果指定 auto_mount_task，自动 mount。
    def create_terminal(self, auto_mount_task: str = None, is_bound: bool = False) -> str:
        short_id = f"term_{uuid.uuid4().hex[:4]}"
        with self._lock:
            while short_id in self._global_terminals:
                short_id = f"term_{uuid.uuid4().hex[:4]}"
            self._global_terminals[short_id] = {
                "created_at": time.time(),
                "is_bound": is_bound,
            }

        session = manager.get_or_create_terminal(short_id)

        if auto_mount_task:
            with self._lock:
                if auto_mount_task not in self._task_mounts:
                    self._task_mounts[auto_mount_task] = set()
                self._task_mounts[auto_mount_task].add(short_id)

        return json.dumps({
            "session_id": short_id,
            "msg": f"New terminal '{short_id}' created (is_bound={is_bound}).",
            "status": session.get_status()
        }, ensure_ascii=False)
    # DOC-END id=session_history/create_terminal#2

    # DOC-BEGIN id=session_history/is_terminal_bound#1 type=query v=1
    # summary: 查询 terminal 的 is_bound 固有属性
    # intent: llm_handlers 中 system_prompt 需要通过此方法过滤 bound terminal，
    #   不将其屏幕快照注入 LLM 上下文。
    def is_terminal_bound(self, session_id: str) -> bool:
        with self._lock:
            meta = self._global_terminals.get(session_id)
            return meta.get("is_bound", False) if meta else False
    # DOC-END id=session_history/is_terminal_bound#1

    # DOC-BEGIN id=session_history/toggle_terminal_bound#1 type=behavior v=1
    # summary: 切换 terminal 的 is_bound 属性，返回新状态
    # intent: 前端 TerminalOverlay 中用户通过 🔒/🔓 图标切换。
    #   切换前不检查是否在 workspace 中使用——允许灵活操作，但用户应自己注意。
    def toggle_terminal_bound(self, session_id: str) -> str:
        with self._lock:
            meta = self._global_terminals.get(session_id)
            if not meta:
                return json.dumps({"ok": False, "error": f"Terminal '{session_id}' not found"})
            meta["is_bound"] = not meta.get("is_bound", False)
            new_val = meta["is_bound"]
        return json.dumps({"ok": True, "session_id": session_id, "is_bound": new_val})
    # DOC-END id=session_history/toggle_terminal_bound#1
    # DOC-BEGIN id=session_history/close_terminal_global#1 type=core v=2
    # summary: 关闭并销毁一个全局 terminal，同时从所有 task mount 列表和 workspace 引用中清理
    # intent: 关闭 terminal 是全局操作，需要清理所有引用：
    #   1. 从 _global_terminals 中删除
    #   2. 从所有 task 的 _task_mounts 中移除
    #   3. 如果某个 workspace 引用了该 terminal（bound 或 interactive），需要清理该 workspace
    #      以及所有关联该 workspace 的 task 的 summary_state 和 rm_index
    #   4. 调用 terminal.py manager 释放 PTY 资源
    def close_terminal(self, session_id: str) -> str:
        with self._lock:
            if session_id not in self._global_terminals:
                return json.dumps({"ok": False, "error": f"Terminal '{session_id}' not found"})
            del self._global_terminals[session_id]
            for task_id, mounts in self._task_mounts.items():
                mounts.discard(session_id)
            # 清理引用该 terminal 的 workspace
            for ws_id in list(self._workspaces.keys()):
                ws = self._workspaces[ws_id]
                if ws.get("bound_terminal_id") == session_id or ws.get("interactive_terminal_id") == session_id:
                    del self._workspaces[ws_id]
                    self._rm_indexes.pop(ws_id, None)
                    # 清除所有关联此 workspace 的 task
                    for tid in list(self._task_workspace.keys()):
                        if self._task_workspace[tid] == ws_id:
                            del self._task_workspace[tid]
                            self._summary_states.pop(tid, None)
        manager.close_session(session_id)
        return json.dumps({"ok": True, "msg": f"Terminal '{session_id}' closed."})
    # DOC-END id=session_history/close_terminal_global#1

    # DOC-BEGIN id=session_history/list_all_terminals#2 type=query v=2
    # summary: 列出所有全局 terminal，包含 is_bound、mounted_tasks、in_workspaces 信息
    # intent: 前端 Terminal 面板和 WorkspaceOverlay 创建表单都需要此信息。
    #   in_workspaces 列出该 terminal 被哪些 workspace 引用，帮助用户避免冲突操作。
    def list_all_terminals(self) -> List[dict]:
        with self._lock:
            result = []
            for sid, meta in self._global_terminals.items():
                mounted_tasks = [
                    tid for tid, mounts in self._task_mounts.items() if sid in mounts
                ]
                in_workspaces = [
                    ws_id for ws_id, ws in self._workspaces.items()
                    if ws.get("bound_terminal_id") == sid or ws.get("interactive_terminal_id") == sid
                ]
                result.append({
                    "session_id": sid,
                    "created_at": meta.get("created_at"),
                    "is_bound": meta.get("is_bound", False),
                    "mounted_tasks": mounted_tasks,
                    "in_workspaces": in_workspaces,
                })
            return result
    # DOC-END id=session_history/list_all_terminals#2

    # === Task Mount 操作 ===

    # DOC-BEGIN id=session_history/mount_terminal#1 type=behavior v=1
    # summary: 将一个全局 terminal 挂载到指定 task
    # intent: mount 后该 terminal 的屏幕快照会出现在该 task 的 LLM system prompt 中
    #   （除非它同时是该 task 的 workspace bind terminal）。
    #   一个 terminal 可以被多个 task 同时 mount。
    def mount_terminal(self, task_id: str, session_id: str) -> str:
        with self._lock:
            if session_id not in self._global_terminals:
                return json.dumps({"ok": False, "error": f"Terminal '{session_id}' not found"})
            if task_id not in self._task_mounts:
                self._task_mounts[task_id] = set()
            self._task_mounts[task_id].add(session_id)
        return json.dumps({"ok": True, "msg": f"Terminal '{session_id}' mounted to task '{task_id}'"})
    # DOC-END id=session_history/mount_terminal#1

    # DOC-BEGIN id=session_history/unmount_terminal#2 type=behavior v=2
    # summary: 从指定 task 卸载一个 terminal
    # intent: unmount 后该 terminal 不再出现在该 task 的 LLM 上下文中。
    #   不再自动解绑 workspace——workspace 是独立实体，unmount terminal 不影响 workspace 关联。
    def unmount_terminal(self, task_id: str, session_id: str) -> str:
        with self._lock:
            if task_id in self._task_mounts:
                self._task_mounts[task_id].discard(session_id)
        return json.dumps({"ok": True, "msg": f"Terminal '{session_id}' unmounted from task '{task_id}'"})
    # DOC-END id=session_history/unmount_terminal#2

    # DOC-BEGIN id=session_history/list_task_mounted#1 type=query v=1
    # summary: 列出指定 task 已挂载的所有 terminal session_id
    # intent: 前端"只看当前任务"过滤 + LLM system prompt 注入都需要此列表。
    def list_task_sessions(self, task_id: str) -> List[str]:
        with self._lock:
            return list(self._task_mounts.get(task_id, set()))
    # DOC-END id=session_history/list_task_mounted#1

    # === Terminal I/O（全局操作，不需要 task_id）===

    # DOC-BEGIN id=session_history/get_session#1 type=internal v=1
    # summary: 根据 session_id 获取 TerminalSession 实例，如果不存在返回 None
    # intent: 全局化后 terminal 直接用 session_id 作为 key（不再需要 task_id 拼接 global_id）。
    #   返回 None 而非自动创建，避免误操作创建幽灵终端。
    def _get_session(self, session_id: str) -> Optional[TerminalSession]:
        return manager.get_session(session_id)
    # DOC-END id=session_history/get_session#1

    def send_to_session(self, task_id: str, session_id: str, data: str) -> str:
        """
        动作函数：向指定进程发送指令。
        task_id 参数保留用于权限检查（未来可验证 terminal 是否 mount 到该 task），
        当前不做校验。
        """
        session = self._get_session(session_id)
        if not session:
            return f"Error: Terminal '{session_id}' not found."
        session.send(data)
        time.sleep(0.3)
        return session.get_status()

    def get_session_status(self, task_id: str, session_id: str) -> str:
        """观察函数：获取指定 terminal 当前屏幕内容。"""
        session = self._get_session(session_id)
        if not session:
            return f"Error: Terminal '{session_id}' not found."
        return session.get_status()
    
    def stage_unsafe_command(self, task_id: str, session_id: str, command: str, reason: str) -> str:
        """
        FC 函数：由 LLM 调用。将命令挂起，不再立即执行。
        返回一条特殊消息，告诉 LLM 命令已进入审批流。
        """
        with self._lock:
            # 每个 task 只允许挂起一个命令
            self._pending_actions[task_id] = {
                "session_id": session_id,
                "command": command,
                "reason": reason,
                "timestamp": time.time()
            }
        
        return json.dumps({
            "status": "staged",
            "msg": f"Command staged for manual approval. The user will review: {command}",
            "instruction": "Please wait for the user to approve and execute this command."
        }, ensure_ascii=False)
    
    def pop_pending_command(self, task_id: str) -> Optional[dict]:
        with self._lock:
            return self._pending_actions.pop(task_id, None)

    def get_pending_command(self, task_id: str) -> Optional[dict]:
        """供前端/管理界面调用，查看当前任务是否有挂起的命令"""
        with self._lock:
            return self._pending_actions.get(task_id)

    def execute_pending_command(self, task_id: str) -> str:
        """
        供人类点击“运行”时调用。
        执行暂存的命令，并清除挂起状态。
        """
        action = None
        with self._lock:
            action = self._pending_actions.pop(task_id, None)

        if not action:
            return "Error: No pending command found for this task."

        # 复用你原有的 send_to_session 逻辑
        # 注意：这里我们直接把命令发给对应的 session
        return self.send_to_session(task_id, action["session_id"], action["command"] + "\n")

    def discard_pending_command(self, task_id: str) -> str:
        """供人类点击“拒绝”时调用"""
        with self._lock:
            self._pending_actions.pop(task_id, None)
        return "Pending command discarded."

    # 修改原有逻辑，注入 System Prompt 状态
    def get_task_context_for_llm(self, task_id: str) -> str:
        """用于在每一轮生成 System Prompt 时调用，告知 LLM 当前挂起状态"""
        action = self.get_pending_command(task_id)
        if action:
            return f"\n[URGENT] There is a PENDING command: `{action['command']}` in session `{action['session_id']}`. Reason: {action['reason']}. It is waiting for user approval. DO NOT suggest new commands until this is resolved."
        return "\nNo pending commands."

    # === Workspace 一等公民管理 ===

    # DOC-BEGIN id=session_history/create_workspace#2 type=core v=2
    # summary: 创建独立 workspace 实体——自动创建两个新 terminal（bound + interactive），
    #   后台线程异步完成 pwd 检测 + .rm/ 扫描，最终状态 creating_terminals → binding → bound/error
    # intent: 前端不再负责选择/创建 terminal，后端全权管理。创建后 terminal 不自动 mount 到任何 task，
    #   workspace 也不自动 attach。用户通过 WorkspaceOverlay 手动 attach。
    #   后台线程依次：创建 bound terminal（is_bound=True）→ 创建 interactive terminal →
    #   pwd 检测 → .rm/ 扫描。任何步骤失败都标记 error 并记录 message。
    def create_workspace(self) -> str:
        ws_id = f"ws_{uuid.uuid4().hex[:6]}"
        with self._lock:
            self._workspaces[ws_id] = {
                "workspace_id": ws_id,
                "bound_terminal_id": None,
                "interactive_terminal_id": None,
                "status": "creating_terminals",
                "message": "Creating terminals...",
                "work_dir": None,
                "created_at": time.time(),
            }

        def _create_and_bind_thread():
            try:
                # Step 1: 创建 bound terminal
                bound_id = self._create_terminal_internal(is_bound=True)
                with self._lock:
                    self._workspaces[ws_id]["bound_terminal_id"] = bound_id
                    self._workspaces[ws_id]["message"] = "Bound terminal created, creating interactive..."

                # Step 2: 创建 interactive terminal
                inter_id = self._create_terminal_internal(is_bound=False)
                with self._lock:
                    self._workspaces[ws_id]["interactive_terminal_id"] = inter_id
                    self._workspaces[ws_id]["status"] = "binding"
                    self._workspaces[ws_id]["message"] = "Detecting working directory..."

                # Step 3: pwd 获取真实工作目录
                self._bind_workspace_phase(ws_id, bound_id)

            except Exception as e:
                logger.error(f"Workspace creation failed for ws={ws_id}: {e}")
                with self._lock:
                    self._workspaces[ws_id]["status"] = "error"
                    self._workspaces[ws_id]["message"] = str(e)

        thread = threading.Thread(target=_create_and_bind_thread, daemon=True)
        thread.start()

        return json.dumps({
            "ok": True,
            "workspace_id": ws_id,
            "status": "creating_terminals",
            "msg": f"Workspace '{ws_id}' creation started",
        }, ensure_ascii=False)
    # DOC-END id=session_history/create_workspace#2

    # DOC-BEGIN id=session_history/create_terminal_internal#1 type=internal v=1
    # summary: 内部方法——创建一个全局 terminal 并注册到 _global_terminals，返回 session_id
    # intent: workspace_create 和 workspace_create_from_recording 都需要在后台线程中创建 terminal，
    #   但不需要 auto_mount_task、不需要返回 JSON 字符串。提取为内部方法避免代码重复。
    #   使用 manager.get_or_create_terminal 确保 PTY 进程真正启动。
    def _create_terminal_internal(self, is_bound: bool = False) -> str:
        short_id = f"term_{uuid.uuid4().hex[:4]}"
        with self._lock:
            while short_id in self._global_terminals:
                short_id = f"term_{uuid.uuid4().hex[:4]}"
            self._global_terminals[short_id] = {
                "created_at": time.time(),
                "is_bound": is_bound,
            }
        manager.get_or_create_terminal(short_id)
        return short_id
    # DOC-END id=session_history/create_terminal_internal#1

    # DOC-BEGIN id=session_history/bind_workspace_phase#1 type=internal v=1
    # summary: 内部方法——workspace 的 binding 阶段：pwd 检测 + RMIndex 加载/构建，
    #   成功后设置 status=bound，失败设置 status=error
    # intent: 从 create_workspace 和 workspace_create_from_recording 中提取公共的 binding 逻辑。
    #   调用前 workspace 必须已有 bound_terminal_id 且 status 已设为 binding。
    #   此方法在后台线程中同步执行，内部通过 _lock 更新 workspace 状态。
    def _bind_workspace_phase(self, ws_id: str, bound_terminal_id: str):
        print(f"[BIND_PHASE][{ws_id}] Starting bind phase with bound_terminal={bound_terminal_id}")

        print(f"[BIND_PHASE][{ws_id}] Running 'pwd' to detect working directory...")
        pwd_result = self.exec_command_reliable(
            task_id="", command="pwd", timeout=10,
            bound_terminal_id=bound_terminal_id,
        )
        print(f"[BIND_PHASE][{ws_id}] pwd result: ok={pwd_result.get('ok')}, stdout={repr(pwd_result.get('stdout', '')[:200])}")
        if pwd_result["ok"] and pwd_result.get("stdout", "").strip():
            detected_dir = pwd_result["stdout"].strip()
        else:
            detected_dir = "/root"
            print(f"[BIND_PHASE][{ws_id}] pwd failed or empty, defaulting to /root")

        print(f"[BIND_PHASE][{ws_id}] Detected work_dir: {detected_dir}")
        with self._lock:
            self._workspaces[ws_id]["work_dir"] = detected_dir
            self._workspaces[ws_id]["message"] = "Scanning .rm/ index..."

        from src.services.rm_index import RMIndex
        rm_index = RMIndex(
            workspace_id=ws_id,
            work_dir=detected_dir,
            history_mgr=self,
            bound_terminal_id=bound_terminal_id,
        )

        print(f"[BIND_PHASE][{ws_id}] Checking .rm/ existence...")
        rm_exists, summary_exists = rm_index.check_rm_exists()
        print(f"[BIND_PHASE][{ws_id}] .rm exists={rm_exists}, summary.json exists={summary_exists}")

        if summary_exists:
            print(f"[BIND_PHASE][{ws_id}] Loading existing .rm/summary.json...")
            with self._lock:
                self._workspaces[ws_id]["message"] = "Loading existing .rm/summary.json..."
            rm_index.load_summary()
            print(f"[BIND_PHASE][{ws_id}] Summary loaded: {len(rm_index.dir_summary)} entries")
        else:
            print(f"[BIND_PHASE][{ws_id}] Building .rm/ index (first time)...")
            with self._lock:
                self._workspaces[ws_id]["message"] = "Building .rm/ index (first time)..."
            rm_index.build_and_save_summary()
            print(f"[BIND_PHASE][{ws_id}] Summary built: {len(rm_index.dir_summary)} entries")

        with self._lock:
            self._rm_indexes[ws_id] = rm_index
            self._workspaces[ws_id]["status"] = "bound"
            self._workspaces[ws_id]["message"] = "Workspace ready"

        print(f"[BIND_PHASE][{ws_id}] Bind phase complete. Status=bound, dir={detected_dir}")
        logger.info(f"Workspace bound: ws={ws_id}, dir={detected_dir}")
    # DOC-END id=session_history/bind_workspace_phase#1

    # DOC-BEGIN id=session_history/workspace_create_from_recording#1 type=core v=1
    # summary: 一站式从录制创建 workspace——后台线程依次完成：创建双 terminal → replay(wait) → binding，
    #   workspace 的 status/message 实时反映进度（creating_terminals → replaying → binding → bound/error）
    # intent: 将原来前端编排的多步流程（创建 terminal → toggle bound → replay → 创建 workspace → 轮询）
    #   完全下沉到后端单线程顺序执行。前端只需发一次请求，之后通过 workspace_info 查看进度。
    #   terminal 不 auto_mount 到任何 task，workspace 不 auto_attach——用户手动操作。
    #   replay 使用 wait=True 同步等待完成，确保 replay 结束后 terminal 处于正确工作目录，
    #   再进行 pwd 检测和 .rm 扫描。任何步骤失败都标记 error 并保留已创建的资源（不回滚删除 terminal）。
    def workspace_create_from_recording(self, recording_id: str, speed_factor: float = 1.0) -> str:
        # 先校验录制是否存在
        rec_result = self.get_recording(recording_id)
        if not rec_result.get("ok"):
            return json.dumps(rec_result)
        rec = rec_result["recording"]

        print(f"[WS_CREATE_FROM_REC] === Starting workspace creation from recording '{recording_id}' ===")
        print(f"[WS_CREATE_FROM_REC] Recording type: {rec.get('type')}")
        print(f"[WS_CREATE_FROM_REC] Recording name: {rec.get('name')}")
        if rec.get("type") == "workspace":
            print(f"[WS_CREATE_FROM_REC] bound_commands count: {len(rec.get('bound_commands', []))}")
            print(f"[WS_CREATE_FROM_REC] interactive_commands count: {len(rec.get('interactive_commands', []))}")
            if rec.get('bound_commands'):
                for i, cmd in enumerate(rec['bound_commands'][:5]):
                    print(f"[WS_CREATE_FROM_REC]   bound_cmd[{i}]: elapsed={cmd.get('elapsed')}, data={repr(cmd.get('data','')[:80])}")
                if len(rec.get('bound_commands', [])) > 5:
                    print(f"[WS_CREATE_FROM_REC]   ... and {len(rec['bound_commands']) - 5} more bound commands")
            if rec.get('interactive_commands'):
                for i, cmd in enumerate(rec['interactive_commands'][:5]):
                    print(f"[WS_CREATE_FROM_REC]   inter_cmd[{i}]: elapsed={cmd.get('elapsed')}, data={repr(cmd.get('data','')[:80])}")
                if len(rec.get('interactive_commands', [])) > 5:
                    print(f"[WS_CREATE_FROM_REC]   ... and {len(rec['interactive_commands']) - 5} more interactive commands")
        elif rec.get("type") == "terminal":
            print(f"[WS_CREATE_FROM_REC] commands count: {len(rec.get('commands', []))}")

        ws_id = f"ws_{uuid.uuid4().hex[:6]}"
        print(f"[WS_CREATE_FROM_REC] Assigned workspace_id: {ws_id}")
        with self._lock:
            self._workspaces[ws_id] = {
                "workspace_id": ws_id,
                "bound_terminal_id": None,
                "interactive_terminal_id": None,
                "status": "creating_terminals",
                "message": "Creating terminals...",
                "work_dir": None,
                "created_at": time.time(),
                "recording_id": recording_id,
            }

        def _full_create_thread():
            try:
                # Step 1: 创建 bound terminal (is_bound=True)
                print(f"[WS_CREATE_FROM_REC][{ws_id}] Step 1: Creating bound terminal (is_bound=True)...")
                bound_id = self._create_terminal_internal(is_bound=True)
                print(f"[WS_CREATE_FROM_REC][{ws_id}] Step 1 done: bound_terminal_id={bound_id}")
                with self._lock:
                    self._workspaces[ws_id]["bound_terminal_id"] = bound_id
                    self._workspaces[ws_id]["message"] = "Bound terminal created, creating interactive..."

                # Step 2: 创建 interactive terminal (is_bound=False)
                print(f"[WS_CREATE_FROM_REC][{ws_id}] Step 2: Creating interactive terminal (is_bound=False)...")
                inter_id = self._create_terminal_internal(is_bound=False)
                print(f"[WS_CREATE_FROM_REC][{ws_id}] Step 2 done: interactive_terminal_id={inter_id}")
                with self._lock:
                    self._workspaces[ws_id]["interactive_terminal_id"] = inter_id
                    self._workspaces[ws_id]["status"] = "replaying"
                    self._workspaces[ws_id]["message"] = "Replaying recorded commands..."

                # Step 3: 回放命令（同步等待完成）
                # DOC-BEGIN id=session_history/ws_create_from_rec/step3_replay#1 type=behavior v=1
                # summary: 调用 replay_recording 并传入 _skip_ws_restore=True，阻止内部 _restore_workspaces 回调将状态提前设为 bound
                # intent: workspace_create_from_recording 自己管理 workspace 状态生命周期
                #   （creating_terminals → replaying → binding → bound），replay 完成后还需要进入 binding 阶段。
                #   如果让 replay_recording 内部的 _restore_workspaces 把状态恢复为 bound，
                #   会导致短暂的状态抖动（bound → binding），且前端轮询可能在这个窗口误判为已就绪。
                print(f"[WS_CREATE_FROM_REC][{ws_id}] Step 3: Replaying commands (wait=True, speed_factor={speed_factor})...")
                if rec["type"] == "workspace":
                    print(f"[WS_CREATE_FROM_REC][{ws_id}]   replay_recording(recording_id={recording_id}, target_bound_tid={bound_id}, target_inter_tid={inter_id})")
                    replay_result = self.replay_recording(
                        recording_id,
                        target_bound_tid=bound_id,
                        target_inter_tid=inter_id,
                        speed_factor=speed_factor,
                        wait=True,
                        _skip_ws_restore=True,
                    )
                elif rec["type"] == "terminal":
                    print(f"[WS_CREATE_FROM_REC][{ws_id}]   replay_recording(recording_id={recording_id}, target_session_id={inter_id})")
                    replay_result = self.replay_recording(
                        recording_id,
                        target_session_id=inter_id,
                        speed_factor=speed_factor,
                        wait=True,
                        _skip_ws_restore=True,
                    )
                # DOC-END id=session_history/ws_create_from_rec/step3_replay#1

                print(f"[WS_CREATE_FROM_REC][{ws_id}] Step 3 result: {replay_result}")

                if not replay_result.get("ok"):
                    raise RuntimeError(f"Replay failed: {replay_result.get('error', str(replay_result))}")

                print(f"[WS_CREATE_FROM_REC][{ws_id}] Step 3 done: replay complete")
                with self._lock:
                    self._workspaces[ws_id]["status"] = "binding"
                    self._workspaces[ws_id]["message"] = "Replay complete, detecting working directory..."

                # Step 4: binding 阶段（pwd + .rm 扫描）
                print(f"[WS_CREATE_FROM_REC][{ws_id}] Step 4: Binding phase (pwd + .rm scan)...")
                self._bind_workspace_phase(ws_id, bound_id)
                print(f"[WS_CREATE_FROM_REC][{ws_id}] Step 4 done: binding complete")

                with self._lock:
                    final_status = self._workspaces.get(ws_id, {}).get("status")
                print(f"[WS_CREATE_FROM_REC][{ws_id}] === Workspace creation finished. Final status: {final_status} ===")

            except Exception as e:
                import traceback
                print(f"[WS_CREATE_FROM_REC][{ws_id}] !!! EXCEPTION: {e}")
                traceback.print_exc()
                logger.error(f"Workspace creation from recording failed for ws={ws_id}: {e}")
                with self._lock:
                    self._workspaces[ws_id]["status"] = "error"
                    self._workspaces[ws_id]["message"] = str(e)

        thread = threading.Thread(target=_full_create_thread, daemon=True)
        thread.start()

        return json.dumps({
            "ok": True,
            "workspace_id": ws_id,
            "status": "creating_terminals",
            "msg": f"Workspace '{ws_id}' creation from recording '{recording_id}' started",
        }, ensure_ascii=False)
    # DOC-END id=session_history/workspace_create_from_recording#1

    # DOC-BEGIN id=session_history/delete_workspace#1 type=behavior v=1
    # summary: 删除 workspace——清除所有 task 的关联、内存缓存，不关闭 terminal
    # intent: 删除是逻辑操作，底层 terminal 继续存在。所有关联此 workspace 的 task 自动解除关联。
    def delete_workspace(self, workspace_id: str) -> str:
        with self._lock:
            if workspace_id not in self._workspaces:
                return json.dumps({"ok": False, "error": f"Workspace '{workspace_id}' not found"})
            del self._workspaces[workspace_id]
            self._rm_indexes.pop(workspace_id, None)
            # 清除所有 task 对此 workspace 的关联
            for task_id in list(self._task_workspace.keys()):
                if self._task_workspace[task_id] == workspace_id:
                    del self._task_workspace[task_id]
                    self._summary_states.pop(task_id, None)
        return json.dumps({"ok": True, "msg": f"Workspace '{workspace_id}' deleted"})
    # DOC-END id=session_history/delete_workspace#1

    # DOC-BEGIN id=session_history/list_workspaces#1 type=query v=1
    # summary: 列出所有 workspace，包含状态、关联的 task 列表
    # intent: 前端 WorkspaceOverlay 需要全量展示。attached_tasks 通过反查 _task_workspace 获得。
    def list_workspaces(self) -> List[dict]:
        with self._lock:
            result = []
            for ws_id, ws in self._workspaces.items():
                attached_tasks = [
                    tid for tid, wid in self._task_workspace.items() if wid == ws_id
                ]
                result.append({
                    **ws,
                    "attached_tasks": attached_tasks,
                })
            return result
    # DOC-END id=session_history/list_workspaces#1

    # DOC-BEGIN id=session_history/attach_workspace#1 type=behavior v=1
    # summary: 将 workspace 关联到 task，自动 mount interactive terminal，创建 SummaryStateManager
    # intent: attach 后 task 获得：
    #   1. 文件操作能力（通过 workspace 的 bound terminal）
    #   2. interactive terminal 自动 mount（屏幕快照注入 LLM）
    #   3. SummaryStateManager 用于 summary 导航
    #   每个 task 最多关联一个 workspace，重复 attach 会自动 detach 旧的。
    # DOC-BEGIN id=session_history/attach_workspace_status_check#1 type=behavior v=1
    # summary: attach_workspace 校验 workspace 状态——仅 status=bound 时允许 attach，
    #   replaying 状态给出专门的友好错误提示
    # intent: replaying 状态说明 replay 尚未完成，terminal 环境不完整，此时 attach 会导致
    #   LLM 读到半成品状态。单独区分 replaying 的报错信息，让前端/用户明确知道需要等待。
    def attach_workspace(self, task_id: str, workspace_id: str) -> str:
        with self._lock:
            ws = self._workspaces.get(workspace_id)
            if not ws:
                return json.dumps({"ok": False, "error": f"Workspace '{workspace_id}' not found"})
            if ws.get("status") == "replaying":
                return json.dumps({"ok": False, "error": f"Workspace '{workspace_id}' is currently replaying. Please wait for replay to finish before attaching."})
            if ws.get("status") != "bound":
                return json.dumps({"ok": False, "error": f"Workspace '{workspace_id}' is not ready (status={ws.get('status')})"})
    # DOC-END id=session_history/attach_workspace_status_check#1

            # 如果已有旧的关联，先清理
            old_ws_id = self._task_workspace.get(task_id)
            if old_ws_id:
                self._summary_states.pop(task_id, None)

            self._task_workspace[task_id] = workspace_id

            # 自动 mount interactive terminal
            inter_tid = ws.get("interactive_terminal_id")
            if inter_tid:
                if task_id not in self._task_mounts:
                    self._task_mounts[task_id] = set()
                self._task_mounts[task_id].add(inter_tid)

            # DOC-BEGIN id=session_history/attach_workspace/create_summary_state#1 type=behavior v=1
            # summary: 创建 SummaryStateManager 时传入 rm_index 和 workspace_id，然后尝试从远程加载已持久化的展开状态
            # intent: rm_index 引用让 SummaryStateManager 能通过 _exec + base64 将状态写入远程 .rm/summary_state.json。
            #   加载必须在 _lock 外执行（load_from_remote 内部会调用 rm_index._exec 做远程 I/O，可能阻塞数秒）。
            #   加载失败不影响功能——SummaryStateManager 以空状态启动即可。
            from src.services.summary_state import SummaryStateManager
            rm_index = self._rm_indexes.get(workspace_id)
            ssm = SummaryStateManager(max_chars=30000, rm_index=rm_index, workspace_id=workspace_id)
            self._summary_states[task_id] = ssm

        # 在锁外执行远程加载
        load_result = ssm.load_from_remote()
        if load_result.get("ok"):
            logger.info(f"Summary state loaded for task '{task_id}': {load_result}")
        else:
            logger.warning(f"Summary state load failed for task '{task_id}': {load_result}")
        # DOC-END id=session_history/attach_workspace/create_summary_state#1

        return json.dumps({"ok": True, "msg": f"Workspace '{workspace_id}' attached to task '{task_id}'"})
    # DOC-END id=session_history/attach_workspace#1

    # DOC-BEGIN id=session_history/detach_workspace#1 type=behavior v=1
    # summary: 解除 task 与 workspace 的关联，清除 summary 状态
    # intent: detach 不影响 workspace 本身（可以被其他 task 或重新 attach）。
    #   不自动 unmount interactive terminal——用户可能仍想保留挂载。
    def detach_workspace(self, task_id: str) -> str:
        with self._lock:
            ws_id = self._task_workspace.pop(task_id, None)
            self._summary_states.pop(task_id, None)
            if not ws_id:
                return json.dumps({"ok": False, "error": f"Task '{task_id}' has no attached workspace"})
        return json.dumps({"ok": True, "msg": f"Workspace '{ws_id}' detached from task '{task_id}'"})
    # DOC-END id=session_history/detach_workspace#1

    # DOC-BEGIN id=session_history/get_workspace_info#2 type=query v=2
    # summary: 获取 workspace 信息——支持通过 task_id（查关联）或 workspace_id（直接查）获取
    # intent: 前端 WorkspaceOverlay 的 workspace_info action 同时传 task_id 和 workspace_id，
    #   优先用 task_id 查找关联的 workspace（常见场景），workspace_id 作为直接查询的备选。
    def get_workspace_info(self, task_id: str = None, workspace_id: str = None) -> Optional[dict]:
        with self._lock:
            ws_id = workspace_id
            if task_id and not ws_id:
                ws_id = self._task_workspace.get(task_id)
            if not ws_id:
                return None
            ws = self._workspaces.get(ws_id)
            if not ws:
                return None
            attached_tasks = [
                tid for tid, wid in self._task_workspace.items() if wid == ws_id
            ]
            return {**ws, "attached_tasks": attached_tasks}
    # DOC-END id=session_history/get_workspace_info#2

    # DOC-BEGIN id=session_history/get_rm_index#2 type=query v=2
    # summary: 获取指定 task 关联的 workspace 的 RMIndex 实例
    # intent: 通过 task → workspace → RMIndex 的链式查找。如果 task 未关联 workspace 返回 None。
    def get_rm_index(self, task_id: str):
        with self._lock:
            ws_id = self._task_workspace.get(task_id)
            if not ws_id:
                return None
            return self._rm_indexes.get(ws_id)
    # DOC-END id=session_history/get_rm_index#2

    # DOC-BEGIN id=session_history/get_summary_state#2 type=query v=2
    # summary: 获取指定 task 的 SummaryStateManager 实例
    # intent: SummaryStateManager 是 per-task 的（不同 task 对同一 workspace 有不同浏览状态）。
    def get_summary_state(self, task_id: str):
        with self._lock:
            return self._summary_states.get(task_id)
    # DOC-END id=session_history/get_summary_state#2

    # === marker 协议可靠执行 ===

    # DOC-BEGIN id=session_history/exec_command_reliable#2 type=design v=2
    # summary: 通过 marker 包裹 + 轮询屏幕实现可靠的远程命令执行，支持直接指定 terminal
    # intent: 有两种调用方式：
    #   1. 传 task_id → 通过 task→workspace→bound_terminal 查找（llm_handlers 远程文件操作）
    #   2. 传 bound_terminal_id → 直接使用指定 terminal（RMIndex 在 workspace 创建阶段调用）
    #   bound_terminal_id 优先级更高，因为 workspace 创建时 task 关联尚未建立。
    def exec_command_reliable(self, task_id: str, command: str, timeout: float = 30.0,
                              bound_terminal_id: str = None) -> dict:
        session_id = bound_terminal_id
        if not session_id:
            info = self.get_workspace_info(task_id=task_id)
            if not info:
                return {"ok": False, "error": "No workspace bound for this task"}
            session_id = info.get("bound_terminal_id")
        session = self._get_session(session_id)
        if not session:
            return {"ok": False, "error": f"Workspace terminal '{session_id}' not found"}

        marker = uuid.uuid4().hex[:8]
        start_marker = f"<<<START_{marker}>>>"
        end_marker = f"<<<END_{marker}>>>"

        wrapped = (
            f"clear && echo '{start_marker}' && "
            f"({command}) 2>&1; "
            f"echo '<<<EC=$?>>>' && echo '{end_marker}'"
        )

        session.send(wrapped + "\r", record=False)

        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.3)
            raw = session.get_raw_screen()
            if end_marker in raw:
                start_idx = raw.find(start_marker)
                end_idx = raw.find(end_marker)
                if start_idx >= 0 and end_idx > start_idx:
                    body = raw[start_idx + len(start_marker):end_idx].strip()
                    exit_code = -1
                    ec_marker = "<<<EC="
                    ec_pos = body.rfind(ec_marker)
                    if ec_pos >= 0:
                        ec_str = body[ec_pos + len(ec_marker):]
                        ec_str = ec_str.split(">>>")[0]
                        try:
                            exit_code = int(ec_str)
                        except ValueError:
                            pass
                        body = body[:ec_pos].strip()
                    return {"ok": True, "stdout": body, "exit_code": exit_code}
                break

        return {"ok": False, "error": "timeout", "raw_screen": session.get_raw_screen()}
    # DOC-END id=session_history/exec_command_reliable#2

    # === 远程文件操作 ===

    # DOC-BEGIN id=session_history/remote_read_file#1 type=design v=1
    # summary: 分段读取远程文件，每段 800 行，避免超出 pyte 屏幕缓冲区
    # intent: pyte 屏幕高度 1000 行，如果 cat 超过 1000 行的文件，顶部内容会被滚掉。
    #   因此先用 wc -l 拿总行数，再用 sed -n 'start,end p' 分段读取，每段 800 行
    #   （留 200 行给 marker、prompt 等开销）。空文件直接返回空字符串。
    def remote_read_file(self, task_id: str, path: str, chunk_lines: int = 800) -> dict:
        wc_result = self.exec_command_reliable(task_id, f"wc -l < '{path}'")
        if not wc_result["ok"]:
            return wc_result

        try:
            total_lines = int(wc_result["stdout"].strip())
        except ValueError:
            return {"ok": False, "error": f"Cannot parse line count: {wc_result['stdout']}"}

        if total_lines == 0:
            return {"ok": True, "content": "", "total_lines": 0}

        content_parts = []
        start = 1
        while start <= total_lines:
            end = start + chunk_lines - 1
            result = self.exec_command_reliable(task_id, f"sed -n '{start},{end}p' '{path}'")
            if not result["ok"]:
                return result
            content_parts.append(result["stdout"])
            start = end + 1

        return {"ok": True, "content": "\n".join(content_parts), "total_lines": total_lines}
    # DOC-END id=session_history/remote_read_file#1

    # DOC-BEGIN id=session_history/remote_write_file#1 type=design v=1
    # summary: 通过 base64 分段传输 + 解码的方式将文件内容写入远程服务器
    # intent: 直接用 echo "content" > file 的方式写入会被 shell 解释特殊字符（引号、$、反引号、
    #   换行符等），导致写入内容被破坏。使用 base64 编码后内容只含 [A-Za-z0-9+/=\n]，完全 shell 安全。
    #   分段是因为单条 shell 命令受 ARG_MAX 限制（通常 ~2MB），每段 48000 字符 base64
    #   （对应约 36KB 原始数据）远低于该限制，也不会撑爆 pyte 屏幕缓冲区。
    #   写入流程：先创建临时 .b64 文件 → 分段 echo -n append → base64 -d 解码到目标路径 → 清理临时文件。
    #   如果任何中间步骤失败，会尽力清理临时文件后返回错误。
    def remote_write_file(self, task_id: str, path: str, content: str, chunk_size: int = 48000) -> dict:
        import base64

        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        tmp_path = f"/tmp/_rw_{uuid.uuid4().hex[:8]}.b64"

        # DOC-BEGIN id=session_history/remote_write_file/mkdir_parent#1 type=behavior v=1
        # summary: 自动创建目标文件的父目录，避免因目录不存在而写入失败
        # intent: 用户可能写入深层路径如 /a/b/c/file.txt，如果 /a/b/c 不存在则 base64 -d 重定向会失败。
        #   这里用 dirname + mkdir -p 预先确保父目录存在。对于相对路径同样有效。
        dir_result = self.exec_command_reliable(task_id, f"mkdir -p \"$(dirname '{path}')\"")
        if not dir_result["ok"]:
            return dir_result
        # DOC-END id=session_history/remote_write_file/mkdir_parent#1

        # 清空/创建临时文件
        result = self.exec_command_reliable(task_id, f"> '{tmp_path}'")
        if not result["ok"]:
            return result

        # DOC-BEGIN id=session_history/remote_write_file/chunked_append#1 type=behavior v=1
        # summary: 将 base64 编码后的内容分段追加到临时文件
        # intent: 单次 echo 的内容不能超过 shell 命令长度限制。分段追加确保每条命令都在安全范围内。
        #   使用 echo -n 避免多余换行符混入 base64 数据导致解码错误。
        #   任何一段写入失败就立即清理临时文件并返回错误，避免残留垃圾文件。
        for i in range(0, len(encoded), chunk_size):
            chunk = encoded[i:i + chunk_size]
            result = self.exec_command_reliable(task_id, f"echo -n '{chunk}' >> '{tmp_path}'")
            if not result["ok"]:
                self.exec_command_reliable(task_id, f"rm -f '{tmp_path}'")
                return {"ok": False, "error": f"Failed writing chunk at offset {i}: {result.get('error', '')}"}
        # DOC-END id=session_history/remote_write_file/chunked_append#1

        # DOC-BEGIN id=session_history/remote_write_file/decode_and_move#1 type=behavior v=1
        # summary: 将临时 base64 文件解码写入目标路径，然后清理临时文件
        # intent: base64 -d 将纯 ASCII 的 base64 内容还原为原始二进制/文本。使用 && 链接确保
        #   解码成功后才删除临时文件。如果解码失败，临时文件保留便于排查。
        #   最后用 wc -c 获取写入字节数作为返回信息，方便调用方确认。
        decode_result = self.exec_command_reliable(
            task_id,
            f"base64 -d '{tmp_path}' > '{path}' && rm -f '{tmp_path}'"
        )
        if not decode_result["ok"]:
            self.exec_command_reliable(task_id, f"rm -f '{tmp_path}'")
            return {"ok": False, "error": f"base64 decode failed: {decode_result.get('error', '')}"}
        # DOC-END id=session_history/remote_write_file/decode_and_move#1

        # DOC-BEGIN id=session_history/remote_write_file/verify#1 type=behavior v=1
        # summary: 写入后校验文件字节数，确保写入完整
        # intent: base64 解码 + 重定向在极端情况下（磁盘满、权限问题、连接断开）可能只写入部分内容。
        #   通过 wc -c 读取实际写入字节数，与原始 content 的 UTF-8 编码字节数对比。
        #   允许 ±1 字节误差（某些系统 wc -c 对尾部换行的处理不同）。
        #   校验失败不删除已写入的文件（可能部分有用），但返回 warning 标记。
        verify_result = self.exec_command_reliable(task_id, f"wc -c < '{path}'")
        expected_bytes = len(content.encode("utf-8"))
        verified = False
        if verify_result["ok"]:
            try:
                actual_bytes = int(verify_result["stdout"].strip())
                verified = abs(actual_bytes - expected_bytes) <= 1
            except ValueError:
                pass
        # DOC-END id=session_history/remote_write_file/verify#1

        # DOC-BEGIN id=session_history/remote_write_file/invalidate_cache#2 type=behavior v=2
        # summary: 写入成功后清除该文件在 RMIndex 中的 DOC 块缓存
        # intent: 文件内容已变更，旧的 DOC 块索引不再有效。通过 task→workspace→rm_index 查找。
        #   path 可能是绝对路径，需要转为相对路径后传给 invalidate_file。
        rm_index = self.get_rm_index(task_id)
        if rm_index:
            rel_path = path
            if path.startswith("/") and rm_index.work_dir:
                prefix = rm_index.work_dir + "/"
                if path.startswith(prefix):
                    rel_path = path[len(prefix):]
            rm_index.invalidate_file(rel_path)
        # DOC-END id=session_history/remote_write_file/invalidate_cache#2

        result = {"ok": True, "msg": f"File written: {path}", "size": len(content)}
        if not verified:
            result["warning"] = f"Size verification uncertain (expected {expected_bytes} bytes)"
        return result
    # DOC-END id=session_history/remote_write_file#1

    # DOC-BEGIN id=session_history/remote_read_file_binary#1 type=design v=1
    # summary: 通过 bound terminal 读取远程二进制文件——先用 base64 编码，再分段读取 base64 文本，
    #   最后在本地解码还原为 bytes，返回 {"ok": True, "data": bytes, "size": int}
    # intent: 图片等二进制文件无法通过 cat/sed 直接传输（会丢失/破坏非 UTF-8 字节）。
    #   利用 base64 将二进制转为纯 ASCII 文本后，可以复用 exec_command_reliable 的 marker 协议安全传输。
    #   base64 编码后每 76 字符一行，1000 行屏幕 × 800 行安全区 ≈ 60800 字符 ≈ 45KB 原始数据/段。
    #   对于典型图片（几十 KB ~ 几 MB）需要多段读取。先用 wc -c 获取 base64 文件大小确定分段数。
    #   设置 max_size 上限（默认 10MB）防止意外读取超大文件导致内存/传输爆炸。
    #   临时 base64 文件写入 /tmp 并在读取完毕后清理，避免残留。
    def remote_read_file_binary(self, task_id: str, path: str, max_size: int = 10 * 1024 * 1024,
                                 chunk_lines: int = 800, bound_terminal_id: str = None) -> dict:
        import base64 as b64_mod

        # Step 0: 检查文件是否存在及大小
        size_result = self.exec_command_reliable(
            task_id, f"wc -c < '{path}'", bound_terminal_id=bound_terminal_id
        )
        if not size_result["ok"]:
            return {"ok": False, "error": f"File not found or inaccessible: {path}"}
        try:
            file_size = int(size_result["stdout"].strip())
        except ValueError:
            return {"ok": False, "error": f"Cannot parse file size: {size_result['stdout']}"}
        if file_size > max_size:
            return {"ok": False, "error": f"File too large: {file_size} bytes (max {max_size})"}
        if file_size == 0:
            return {"ok": True, "data": b"", "size": 0}

        # Step 1: 在远程生成 base64 临时文件
        import uuid as _uuid
        tmp_b64 = f"/tmp/_rb_{_uuid.uuid4().hex[:8]}.b64"
        encode_result = self.exec_command_reliable(
            task_id, f"base64 '{path}' > '{tmp_b64}'",
            timeout=30.0, bound_terminal_id=bound_terminal_id
        )
        if not encode_result["ok"]:
            self.exec_command_reliable(task_id, f"rm -f '{tmp_b64}'", bound_terminal_id=bound_terminal_id)
            return {"ok": False, "error": f"base64 encode failed: {encode_result.get('error', '')}"}

        # Step 2: 获取 base64 文件行数
        wc_result = self.exec_command_reliable(
            task_id, f"wc -l < '{tmp_b64}'", bound_terminal_id=bound_terminal_id
        )
        if not wc_result["ok"]:
            self.exec_command_reliable(task_id, f"rm -f '{tmp_b64}'", bound_terminal_id=bound_terminal_id)
            return {"ok": False, "error": f"Cannot get line count of base64 file"}
        try:
            total_lines = int(wc_result["stdout"].strip())
        except ValueError:
            self.exec_command_reliable(task_id, f"rm -f '{tmp_b64}'", bound_terminal_id=bound_terminal_id)
            return {"ok": False, "error": f"Cannot parse line count: {wc_result['stdout']}"}

        # Step 3: 分段读取 base64 内容
        b64_parts = []
        start = 1
        while start <= total_lines:
            end = start + chunk_lines - 1
            chunk_result = self.exec_command_reliable(
                task_id, f"sed -n '{start},{end}p' '{tmp_b64}'",
                bound_terminal_id=bound_terminal_id
            )
            if not chunk_result["ok"]:
                self.exec_command_reliable(task_id, f"rm -f '{tmp_b64}'", bound_terminal_id=bound_terminal_id)
                return {"ok": False, "error": f"Failed reading chunk at line {start}: {chunk_result.get('error', '')}"}
            b64_parts.append(chunk_result["stdout"])
            start = end + 1

        # Step 4: 清理远程临时文件
        self.exec_command_reliable(task_id, f"rm -f '{tmp_b64}'", bound_terminal_id=bound_terminal_id)

        # Step 5: 本地 base64 解码
        b64_text = "".join(b64_parts).replace("\n", "").replace("\r", "").replace(" ", "")
        try:
            raw_bytes = b64_mod.b64decode(b64_text)
        except Exception as e:
            return {"ok": False, "error": f"base64 decode failed locally: {e}"}

        return {"ok": True, "data": raw_bytes, "size": len(raw_bytes)}
    # DOC-END id=session_history/remote_read_file_binary#1

    def remote_list_dir(self, task_id: str, path: str) -> dict:
        return self.exec_command_reliable(task_id, f"ls -la '{path}'")

    def remote_file_exists(self, task_id: str, path: str) -> dict:
        result = self.exec_command_reliable(task_id, f"test -f '{path}' && echo 'EXISTS' || echo 'NOT_FOUND'")
        if result["ok"]:
            result["exists"] = result["stdout"].strip() == "EXISTS"
        return result

    def remote_mkdir(self, task_id: str, path: str) -> dict:
        return self.exec_command_reliable(task_id, f"mkdir -p '{path}'")

    # === 命令录制 & 回放 ===

    # DOC-BEGIN id=session_history/get_terminal_command_log#1 type=query v=1
    # summary: 获取指定 terminal 的命令录制日志
    # intent: 前端展示当前 terminal 已录制的命令列表，用户决定是否保存。
    def get_terminal_command_log(self, session_id: str) -> dict:
        session = self._get_session(session_id)
        if not session:
            return {"ok": False, "error": f"Terminal '{session_id}' not found"}
        return {"ok": True, "commands": session.get_command_log(), "session_id": session_id}
    # DOC-END id=session_history/get_terminal_command_log#1

    # DOC-BEGIN id=session_history/clear_terminal_command_log#1 type=behavior v=1
    # summary: 清空指定 terminal 的命令录制日志
    # intent: 用户保存录制后想清空重新开始，或者不需要的命令太多想重置。
    def clear_terminal_command_log(self, session_id: str) -> dict:
        session = self._get_session(session_id)
        if not session:
            return {"ok": False, "error": f"Terminal '{session_id}' not found"}
        session.clear_command_log()
        return {"ok": True, "msg": f"Command log cleared for '{session_id}'"}
    # DOC-END id=session_history/clear_terminal_command_log#1

    # DOC-BEGIN id=session_history/save_terminal_recording#1 type=behavior v=1
    # summary: 将指定 terminal 的命令日志保存为录制文件（JSON），持久化到磁盘
    # intent: 录制文件命名格式 rec_<uuid>.json，包含 type=terminal、用户指定的 name、命令列表。
    #   保存后不自动清空命令日志——用户可能想继续追加录制。
    #   文件存储在 data/recordings/ 目录下。
    def save_terminal_recording(self, session_id: str, name: str = None) -> dict:
        session = self._get_session(session_id)
        if not session:
            return {"ok": False, "error": f"Terminal '{session_id}' not found"}
        commands = session.get_command_log()
        if not commands:
            return {"ok": False, "error": "No commands recorded yet"}

        rec_id = f"rec_{uuid.uuid4().hex[:8]}"
        recording = {
            "id": rec_id,
            "type": "terminal",
            "name": name or f"{session_id} recording",
            "source_session_id": session_id,
            "commands": commands,
            "created_at": time.time(),
        }
        filepath = os.path.join(self._recordings_dir, f"{rec_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(recording, f, ensure_ascii=False, indent=2)

        return {"ok": True, "recording_id": rec_id, "command_count": len(commands), "filepath": filepath}
    # DOC-END id=session_history/save_terminal_recording#1

    # DOC-BEGIN id=session_history/save_workspace_recording#1 type=behavior v=1
    # summary: 保存 workspace 录制——同时保存 bound terminal 和 interactive terminal 的命令日志
    # intent: workspace 录制包含两个 terminal 的命令日志，回放时需要同时驱动两个 terminal。
    #   通过 workspace_id 找到两个 terminal，分别获取各自的命令日志。
    #   如果两个 terminal 都没有命令，拒绝保存。
    def save_workspace_recording(self, workspace_id: str, name: str = None) -> dict:
        with self._lock:
            ws = self._workspaces.get(workspace_id)
            if not ws:
                return {"ok": False, "error": f"Workspace '{workspace_id}' not found"}
            bound_tid = ws.get("bound_terminal_id")
            inter_tid = ws.get("interactive_terminal_id")

        bound_session = self._get_session(bound_tid) if bound_tid else None
        inter_session = self._get_session(inter_tid) if inter_tid else None

        bound_cmds = bound_session.get_command_log() if bound_session else []
        inter_cmds = inter_session.get_command_log() if inter_session else []

        if not bound_cmds and not inter_cmds:
            return {"ok": False, "error": "No commands recorded in either terminal"}

        rec_id = f"rec_{uuid.uuid4().hex[:8]}"
        recording = {
            "id": rec_id,
            "type": "workspace",
            "name": name or f"{workspace_id} recording",
            "source_workspace_id": workspace_id,
            "bound_terminal_id": bound_tid,
            "interactive_terminal_id": inter_tid,
            "bound_commands": bound_cmds,
            "interactive_commands": inter_cmds,
            "created_at": time.time(),
        }
        filepath = os.path.join(self._recordings_dir, f"{rec_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(recording, f, ensure_ascii=False, indent=2)

        return {
            "ok": True, "recording_id": rec_id,
            "bound_count": len(bound_cmds), "interactive_count": len(inter_cmds),
        }
    # DOC-END id=session_history/save_workspace_recording#1

    # DOC-BEGIN id=session_history/list_recordings#1 type=query v=1
    # summary: 列出所有已保存的录制文件（terminal 和 workspace 类型）
    # intent: 从 data/recordings/ 目录读取所有 JSON 文件，返回元信息（不含完整 commands）。
    #   排序按创建时间降序（最新的在前）。
    def list_recordings(self, rec_type: str = None) -> dict:
        recordings = []
        for filename in os.listdir(self._recordings_dir):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(self._recordings_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                if rec_type and rec.get("type") != rec_type:
                    continue
                summary = {
                    "id": rec.get("id"),
                    "type": rec.get("type"),
                    "name": rec.get("name"),
                    "created_at": rec.get("created_at"),
                }
                if rec.get("type") == "terminal":
                    summary["command_count"] = len(rec.get("commands", []))
                    summary["source_session_id"] = rec.get("source_session_id")
                elif rec.get("type") == "workspace":
                    summary["bound_count"] = len(rec.get("bound_commands", []))
                    summary["interactive_count"] = len(rec.get("interactive_commands", []))
                    summary["source_workspace_id"] = rec.get("source_workspace_id")
                recordings.append(summary)
            except Exception:
                continue
        recordings.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        return {"ok": True, "recordings": recordings}
    # DOC-END id=session_history/list_recordings#1

    # DOC-BEGIN id=session_history/get_recording#1 type=query v=1
    # summary: 获取指定录制的完整内容（含所有命令）
    # intent: 前端查看录制详情或准备回放时需要完整数据。
    def get_recording(self, recording_id: str) -> dict:
        filepath = os.path.join(self._recordings_dir, f"{recording_id}.json")
        if not os.path.exists(filepath):
            return {"ok": False, "error": f"Recording '{recording_id}' not found"}
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                rec = json.load(f)
            return {"ok": True, "recording": rec}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    # DOC-END id=session_history/get_recording#1

    # DOC-BEGIN id=session_history/delete_recording#1 type=behavior v=1
    # summary: 删除指定录制文件
    # intent: 用户不再需要某个录制时调用。只删除文件，不影响任何运行中的回放。
    def delete_recording(self, recording_id: str) -> dict:
        filepath = os.path.join(self._recordings_dir, f"{recording_id}.json")
        if not os.path.exists(filepath):
            return {"ok": False, "error": f"Recording '{recording_id}' not found"}
        try:
            os.remove(filepath)
            return {"ok": True, "msg": f"Recording '{recording_id}' deleted"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    # DOC-END id=session_history/delete_recording#1

    # DOC-BEGIN id=session_history/replay_to_terminal#1 type=core v=1
    # summary: 将录制的命令序列回放到指定 terminal——后台线程异步执行，支持中途停止
    # intent: 回放逻辑的核心：遍历命令列表，对每条命令：
    #   1. 等待 elapsed 时间（录制时的间隔），但每 0.3s 检查一次终端是否已回到 prompt
    #   2. 如果终端已 prompt ready，立即发送下一条命令（不等完整间隔）
    #   3. 如果收到 stop 信号，立即终止回放
    #   回放的命令 record=False，不污染当前 terminal 的录制日志。
    #   同一个 terminal 同时只能有一个回放——新回放会先停止旧的。
    #   speed_factor 控制回放速度：1.0=原速，2.0=两倍速，0.5=半速。
    def replay_to_terminal(self, session_id: str, commands: list,
                            speed_factor: float = 1.0) -> dict:
        session = self._get_session(session_id)
        if not session:
            print(f"[REPLAY_TO_TERMINAL] ERROR: Terminal '{session_id}' not found!")
            return {"ok": False, "error": f"Terminal '{session_id}' not found"}

        print(f"[REPLAY_TO_TERMINAL] Starting replay on '{session_id}': {len(commands)} commands, speed_factor={speed_factor}")

        # 停止该 terminal 上已有的回放
        self.stop_replay(session_id)

        stop_event = threading.Event()
        with self._lock:
            self._active_replays[session_id] = stop_event

        def _replay_thread():
            try:
                print(f"[REPLAY_TO_TERMINAL][{session_id}] Replay thread started, {len(commands)} commands to replay")
                for i, cmd_entry in enumerate(commands):
                    if stop_event.is_set():
                        print(f"[REPLAY_TO_TERMINAL][{session_id}] Stop event set at command {i}, aborting")
                        break

                    elapsed = cmd_entry.get("elapsed", 0.5)
                    data = cmd_entry.get("data", "")

                    # DOC-BEGIN id=session_history/replay_wait_logic#1 type=design v=1
                    # summary: 等待回放间隔时间，但如果终端提前回到 prompt 状态则立即发送
                    # intent: 两个条件满足其一即可发送下一条命令：
                    #   (a) 经过了 elapsed / speed_factor 秒（录制时的真实间隔按速度因子缩放）
                    #   (b) 终端 is_prompt_ready() 返回 True（shell 已处于等待输入状态）
                    #   使用 0.3s 粒度轮询兼顾响应速度和 CPU 开销。
                    #   第一条命令（i==0）跳过等待——通常终端已经在 prompt 状态。
                    if i > 0:
                        adjusted_wait = elapsed / speed_factor if speed_factor > 0 else elapsed
                        waited = 0.0
                        while waited < adjusted_wait:
                            if stop_event.is_set():
                                break
                            time.sleep(0.3)
                            waited += 0.3
                            if session.is_prompt_ready():
                                break
                    # DOC-END id=session_history/replay_wait_logic#1

                    if stop_event.is_set():
                        print(f"[REPLAY_TO_TERMINAL][{session_id}] Stop event set before send at cmd {i}, aborting")
                        break

                    if i < 3 or i == len(commands) - 1:
                        print(f"[REPLAY_TO_TERMINAL][{session_id}] Sending cmd[{i}/{len(commands)}]: data={repr(data[:60])}")
                    elif i == 3:
                        print(f"[REPLAY_TO_TERMINAL][{session_id}] ... (suppressing per-command logs for middle commands) ...")

                    session.send(data, record=False)

            except Exception as e:
                print(f"[REPLAY_TO_TERMINAL][{session_id}] !!! Exception in replay thread: {e}")
                import traceback
                traceback.print_exc()
            finally:
                with self._lock:
                    self._active_replays.pop(session_id, None)
                print(f"[REPLAY_TO_TERMINAL][{session_id}] Replay thread finished")
                logger.info(f"Replay finished for terminal '{session_id}'")

        thread = threading.Thread(target=_replay_thread, daemon=True)
        thread.start()
        # DOC-BEGIN id=session_history/replay_to_terminal/return_thread#1 type=behavior v=1
        # summary: 返回结果中附带 _thread 引用，供 replay_recording 的 wait 模式使用
        # intent: 正常 JSON 序列化时 _thread 会被忽略（非可序列化字段），不影响 API 返回。
        #   但 replay_recording 内部可以拿到 thread 对象进行 join 等待。
        return {"ok": True, "msg": f"Replay started on '{session_id}' with {len(commands)} commands",
                "speed_factor": speed_factor, "_thread": thread}
        # DOC-END id=session_history/replay_to_terminal/return_thread#1
    # DOC-END id=session_history/replay_to_terminal#1

    # DOC-BEGIN id=session_history/find_workspace_by_terminal#1 type=query v=1
    # summary: 根据 terminal session_id 反查其归属的 workspace_id（bound 或 interactive 均匹配）
    # intent: replay_recording 需要知道目标 terminal 属于哪个 workspace，以便自动标记 replaying 状态。
    #   一个 terminal 理论上只归属一个 workspace（创建时校验保证），找到第一个即返回。
    #   调用方须持有 _lock 或在无竞争场景下调用。
    def _find_workspace_by_terminal(self, session_id: str) -> Optional[str]:
        for ws_id, ws in self._workspaces.items():
            if ws.get("bound_terminal_id") == session_id or ws.get("interactive_terminal_id") == session_id:
                return ws_id
        return None
    # DOC-END id=session_history/find_workspace_by_terminal#1

    # DOC-BEGIN id=session_history/replay_recording#3 type=core v=3
    # summary: 回放已保存的录制——terminal 类型回放到单个 terminal，workspace 类型同时回放到两个 terminal。
    #   自动检测目标 terminal 归属的 workspace 并标记为 replaying 状态，回放完成后恢复为 bound。
    #   wait=True 时阻塞等待所有回放线程完成后再返回，wait=False（默认）立即返回。
    # intent: 引入 replaying 状态解决"replay 未完成就 attach → LLM 读到不完整环境"的竞态问题。
    #   后端自动管理状态，前端/调用方无需额外调用标记 API。
    #   replay 线程全部完成后（包括异常退出）在 finally 中恢复 bound，确保不会卡在 replaying。
    #   wait 模式用于自动化流程（replay 后紧接 .rm summary 构建），
    #   非 wait 模式用于前端手动操作（通过轮询 workspace_info 观察状态变化）。
    def replay_recording(self, recording_id: str, target_session_id: str = None,
                          target_bound_tid: str = None, target_inter_tid: str = None,
                          speed_factor: float = 1.0, wait: bool = False,
                          _skip_ws_restore: bool = False) -> dict:
        rec_result = self.get_recording(recording_id)
        if not rec_result.get("ok"):
            return rec_result
        rec = rec_result["recording"]

        print(f"[REPLAY_RECORDING] === replay_recording called ===")
        print(f"[REPLAY_RECORDING] recording_id={recording_id}, type={rec.get('type')}")
        print(f"[REPLAY_RECORDING] target_session_id={target_session_id}, target_bound_tid={target_bound_tid}, target_inter_tid={target_inter_tid}")
        print(f"[REPLAY_RECORDING] speed_factor={speed_factor}, wait={wait}")
        if rec.get("type") == "workspace":
            print(f"[REPLAY_RECORDING] bound_commands: {len(rec.get('bound_commands', []))}, interactive_commands: {len(rec.get('interactive_commands', []))}")
        elif rec.get("type") == "terminal":
            print(f"[REPLAY_RECORDING] commands: {len(rec.get('commands', []))}")

        threads_to_wait = []

        # DOC-BEGIN id=session_history/replay_recording/detect_workspace#2 type=behavior v=2
        # summary: 收集所有目标 terminal id，反查归属 workspace 并标记为 replaying
        # intent: 对 bound 和 replaying 状态的 workspace 都进行标记（replaying 覆盖 replaying 是幂等的）。
        #   从 workspace_create_from_recording 调用时，workspace 已经是 replaying 状态，
        #   旧代码只检查 status==bound 会导致跳过，恢复回调 _restore_workspaces 不会执行。
        all_target_tids = set()
        if target_session_id:
            all_target_tids.add(target_session_id)
        if target_bound_tid:
            all_target_tids.add(target_bound_tid)
        if target_inter_tid:
            all_target_tids.add(target_inter_tid)

        affected_ws_ids = set()
        with self._lock:
            for tid in all_target_tids:
                ws_id = self._find_workspace_by_terminal(tid)
                if ws_id:
                    ws_status = self._workspaces.get(ws_id, {}).get("status")
                    print(f"[REPLAY_RECORDING] Terminal '{tid}' belongs to workspace '{ws_id}' (status={ws_status})")
                    if ws_status in ("bound", "replaying"):
                        affected_ws_ids.add(ws_id)
                else:
                    print(f"[REPLAY_RECORDING] Terminal '{tid}' does not belong to any workspace")
            for ws_id in affected_ws_ids:
                self._workspaces[ws_id]["status"] = "replaying"
                self._workspaces[ws_id]["message"] = "Replaying recorded commands..."
        print(f"[REPLAY_RECORDING] affected_ws_ids={affected_ws_ids}")
        # DOC-END id=session_history/replay_recording/detect_workspace#2

        # DOC-BEGIN id=session_history/replay_recording/restore_workspace#1 type=behavior v=1
        # summary: 定义 _restore_workspaces 回调——将所有受影响 workspace 的状态恢复为 bound
        # intent: 此回调在所有 replay 线程完成后调用（无论成功、异常还是被 stop）。
        #   使用闭包捕获 affected_ws_ids，确保只恢复本次 replay 标记的 workspace。
        #   仅恢复当前仍处于 replaying 的 workspace——如果用户在 replay 期间手动删除了 workspace，
        #   该 ws_id 不会存在于 _workspaces 中，安全跳过。
        # DOC-BEGIN id=session_history/replay_recording/restore_workspace_skip#1 type=behavior v=1
        # summary: _restore_workspaces 回调——将受影响 workspace 状态恢复为 bound；
        #   当 _skip_ws_restore=True 时跳过恢复（由调用方自行管理状态）
        # intent: workspace_create_from_recording 调用 replay_recording 时传入 _skip_ws_restore=True，
        #   因为调用方在 replay 后还有 binding 阶段，不应提前恢复为 bound。
        #   普通前端手动 replay 则需要 _restore_workspaces 自动恢复。
        def _restore_workspaces():
            if _skip_ws_restore:
                print(f"[REPLAY_RECORDING] _skip_ws_restore=True, skipping workspace status restore for: {affected_ws_ids}")
                return
            with self._lock:
                for ws_id in affected_ws_ids:
                    ws = self._workspaces.get(ws_id)
                    if ws and ws.get("status") == "replaying":
                        ws["status"] = "bound"
                        ws["message"] = "Workspace ready"
            logger.info(f"Workspace status restored to 'bound' for: {affected_ws_ids}")
        # DOC-END id=session_history/replay_recording/restore_workspace_skip#1
        # DOC-END id=session_history/replay_recording/restore_workspace#1

        if rec["type"] == "terminal":
            if not target_session_id:
                _restore_workspaces()
                return {"ok": False, "error": "target_session_id is required for terminal recording replay"}
            print(f"[REPLAY_RECORDING] Replaying terminal recording to session '{target_session_id}', {len(rec['commands'])} commands")
            result = self.replay_to_terminal(target_session_id, rec["commands"], speed_factor)
            t = result.pop("_thread", None)
            print(f"[REPLAY_RECORDING] replay_to_terminal result: ok={result.get('ok')}, thread={'exists' if t else 'None'}")
            if wait and t:
                threads_to_wait.append(t)
            elif not wait:
                # summary: 非 wait 模式下：有 replay 线程时启动守护线程等待完成后恢复；无线程时立即恢复
                # intent: replay_to_terminal 可能失败（terminal 不存在），此时 t 为 None，
                #   如果不立即恢复，workspace 会永远卡在 replaying 状态。
                if t and affected_ws_ids:
                    def _wait_and_restore():
                        t.join(timeout=3600)
                        _restore_workspaces()
                    threading.Thread(target=_wait_and_restore, daemon=True).start()
                elif affected_ws_ids:
                    _restore_workspaces()
                # DOC-END id=session_history/replay_recording/async_restore_terminal#1
                return result

        elif rec["type"] == "workspace":
            results = {}
            print(f"[REPLAY_RECORDING] Workspace replay: bound_commands={len(rec.get('bound_commands', []))}, interactive_commands={len(rec.get('interactive_commands', []))}")
            print(f"[REPLAY_RECORDING]   target_bound_tid={target_bound_tid}, target_inter_tid={target_inter_tid}")
            if rec.get("bound_commands") and target_bound_tid:
                print(f"[REPLAY_RECORDING]   Starting bound replay: {len(rec['bound_commands'])} commands -> {target_bound_tid}")
                r = self.replay_to_terminal(target_bound_tid, rec["bound_commands"], speed_factor)
                t = r.pop("_thread", None)
                print(f"[REPLAY_RECORDING]   Bound replay result: ok={r.get('ok')}, error={r.get('error', 'none')}, thread={'exists' if t else 'None'}")
                if t:
                    threads_to_wait.append(t)
                results["bound"] = r
            else:
                print(f"[REPLAY_RECORDING]   SKIPPING bound replay: bound_commands={'empty' if not rec.get('bound_commands') else 'exists'}, target_bound_tid={target_bound_tid}")
            if rec.get("interactive_commands") and target_inter_tid:
                print(f"[REPLAY_RECORDING]   Starting interactive replay: {len(rec['interactive_commands'])} commands -> {target_inter_tid}")
                r = self.replay_to_terminal(target_inter_tid, rec["interactive_commands"], speed_factor)
                t = r.pop("_thread", None)
                print(f"[REPLAY_RECORDING]   Interactive replay result: ok={r.get('ok')}, error={r.get('error', 'none')}, thread={'exists' if t else 'None'}")
                if t:
                    threads_to_wait.append(t)
                results["interactive"] = r
            else:
                print(f"[REPLAY_RECORDING]   SKIPPING interactive replay: interactive_commands={'empty' if not rec.get('interactive_commands') else 'exists'}, target_inter_tid={target_inter_tid}")
            print(f"[REPLAY_RECORDING]   threads_to_wait count: {len(threads_to_wait)}")
            if not results:
                print(f"[REPLAY_RECORDING]   !!! No results — nothing was replayed")
                _restore_workspaces()
                return {"ok": False, "error": "No target terminals specified or no commands to replay"}
            if not wait:
                # DOC-BEGIN id=session_history/replay_recording/async_restore_workspace#1 type=behavior v=2
                # summary: 非 wait 模式下：有 replay 线程时启动守护线程等待完成后恢复；无线程时立即恢复
                # intent: 两个 replay_to_terminal 都可能失败（terminal 不存在），
                #   threads_to_wait 为空时必须立即恢复，否则 workspace 永远卡在 replaying。
                if threads_to_wait and affected_ws_ids:
                    def _wait_and_restore():
                        for th in threads_to_wait:
                            th.join(timeout=3600)
                        _restore_workspaces()
                    threading.Thread(target=_wait_and_restore, daemon=True).start()
                elif affected_ws_ids:
                    _restore_workspaces()
                # DOC-END id=session_history/replay_recording/async_restore_workspace#1
                return {"ok": True, "results": results}

        else:
            _restore_workspaces()
            return {"ok": False, "error": f"Unknown recording type: {rec['type']}"}

        # DOC-BEGIN id=session_history/replay_recording/wait_join#2 type=behavior v=2
        # summary: wait=True 时阻塞 join 所有回放线程，完成后恢复 workspace 状态
        # intent: join 超时 3600s 防止永久阻塞。恢复在 join 之后确保 replay 真正完成。
        print(f"[REPLAY_RECORDING] Waiting for {len(threads_to_wait)} replay thread(s) to finish...")
        for i, t in enumerate(threads_to_wait):
            print(f"[REPLAY_RECORDING]   Joining thread {i}...")
            t.join(timeout=3600)
            print(f"[REPLAY_RECORDING]   Thread {i} finished (alive={t.is_alive()})")
        _restore_workspaces()
        print(f"[REPLAY_RECORDING] All threads joined, workspaces restored")
        # DOC-END id=session_history/replay_recording/wait_join#2

        if rec["type"] == "terminal":
            return {"ok": True, "msg": "Replay completed (waited)", "recording_id": recording_id}
        else:
            return {"ok": True, "results": results, "msg": "Replay completed (waited)", "recording_id": recording_id}
    # DOC-END id=session_history/replay_recording#3

    # DOC-BEGIN id=session_history/stop_replay#1 type=behavior v=1
    # summary: 停止指定 terminal 上正在进行的回放
    # intent: 设置 stop_event 后回放线程会在下一个轮询周期（≤0.3s）内停止。
    def stop_replay(self, session_id: str) -> dict:
        with self._lock:
            stop_event = self._active_replays.pop(session_id, None)
        if stop_event:
            stop_event.set()
            return {"ok": True, "msg": f"Replay stopped for '{session_id}'"}
        return {"ok": True, "msg": "No active replay"}
    # DOC-END id=session_history/stop_replay#1

    # DOC-BEGIN id=session_history/get_replay_status#1 type=query v=1
    # summary: 查询哪些 terminal 正在回放中
    # intent: 前端需要知道哪些 terminal 有活跃的回放，以显示停止按钮。
    def get_replay_status(self) -> dict:
        with self._lock:
            active = list(self._active_replays.keys())
        return {"ok": True, "replaying": active}
    # DOC-END id=session_history/get_replay_status#1

# 导出单例供 API 使用
history_manager = TaskHistoryManager()
