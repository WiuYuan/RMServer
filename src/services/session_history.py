import threading
from typing import Dict, List, Optional
from src.services.terminal import manager, TerminalSession
import time
import uuid
import json

class TaskHistoryManager:
    """
    任务历史管理器
    通过 task_id 和 session_id 的双重映射，实现进程级别的唯一标识和存储
    """
    def __init__(self):
        # 结构: { task_id: { session_id: global_terminal_id } }
        self._task_map: Dict[str, Dict[str, str]] = {}
        self._lock = threading.Lock()
        self._pending_actions: Dict[str, Optional[dict]] = {}

    def _get_global_id(self, task_id: str, session_id: str) -> str:
        """生成全局唯一的标识符，防止不同任务间的 session_id 冲突"""
        return f"task_{task_id}_idx_{session_id}"

    def get_or_create_session(self, task_id: str, session_id: str) -> TerminalSession:
        """在特定任务下获取或创建一个进程"""
        global_id = self._get_global_id(task_id, session_id)
        
        with self._lock:
            if task_id not in self._task_map:
                self._task_map[task_id] = {}
            self._task_map[task_id][session_id] = global_id
            
        # 调用 terminal.py 中的单例 manager
        return manager.get_or_create_terminal(global_id)

    def list_task_sessions(self, task_id: str) -> List[str]:
        """查看某个任务下当前运行的所有进程 ID"""
        with self._lock:
            if task_id in self._task_map:
                return list(self._task_map[task_id].keys())
            return []

    def cleanup_task(self, task_id: str):
        """清理一个任务相关的所有终端历史"""
        with self._lock:
            if task_id in self._task_map:
                for session_id, global_id in self._task_map[task_id].items():
                    manager.close_session(global_id)
                del self._task_map[task_id]
        
    def create_random_session(self, task_id: str) -> str:
        """
        生成一个终端。
        返回生成的 session_id 和初始状态。
        """
        # 生成一个简短的唯一 ID，例如 term_a1b2
        short_id = f"term_{uuid.uuid4().hex[:4]}"
        
        # 确保不会撞名（极低概率，但在任务内需保证唯一）
        existing = self.list_task_sessions(task_id)
        while short_id in existing:
            short_id = f"term_{uuid.uuid4().hex[:4]}"
            
        session = self.get_or_create_session(task_id, short_id)
        return json.dumps({
            "session_id": short_id,
            "msg": f"New terminal session '{short_id}' created.",
            "status": session.get_status()
        }, ensure_ascii=False)
        
    def close_terminal(self, task_id: str, session_id: str) -> str:
        """
        终止并销毁指定任务下的特定终端进程。
        """
        with self._lock:
            if task_id in self._task_map and session_id in self._task_map[task_id]:
                global_id = self._task_map[task_id].pop(session_id)
                # 调用 terminal.py 中的单例 manager 彻底关闭 PTY 和进程
                manager.close_session(global_id)
                return f"Terminal session '{session_id}' has been closed and resources released."
            else:
                return f"Error: Terminal session '{session_id}' not found in task '{task_id}'."
                
    def send_to_session(self, task_id: str, session_id: str, data: str) -> str:
        """
        动作函数：向指定任务的指定进程发送指令。
        发送后会等待一小段时间并返回当前屏幕快照。
        """
        # 获取（或自动创建）会话
        session = self.get_or_create_session(task_id, session_id)
        
        # 发送指令
        session.send(data)
        
        # 这里的 sleep 是为了给后台线程一点时间处理 PTY 返回的数据并渲染到 pyte
        # 对于 nano/vim 等 TUI，建议 0.3s-0.5s
        time.sleep(0.3) 
        
        # 返回快照，让 LLM 看到指令执行后的即时反应
        return session.get_status()
    

    def get_session_status(self, task_id: str, session_id: str) -> str:
        """
        观察函数：仅获取指定进程当前的屏幕内容。
        用于 LLM 在决定下一步行动前观察状态（例如检查长耗时任务进度）。
        """
        session = self.get_or_create_session(task_id, session_id)
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

# 导出单例供 API 使用
history_manager = TaskHistoryManager()