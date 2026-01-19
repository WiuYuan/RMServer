import pty
import os
import time
import pyte
import select
import subprocess
import threading
from typing import Dict, Optional, Type, List

# === 基础抽象层 ===

class BaseSession:
    """所有交互式会话的基类，方便后续扩展文件系统或其他系统"""
    def __init__(self, session_id: str):
        self.session_id = session_id

    def send(self, data: str):
        raise NotImplementedError

    def get_status(self) -> str:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

# === 终端实现层 ===

class TerminalSession(BaseSession):
    def __init__(self, session_id: str, width: int = 1000, height: int = 1000, shell: str = "bash"):
        super().__init__(session_id)
        self.width = width
        self.height = height
        
        # 1. 初始化 Pyte 虚拟屏幕
        base_path = "/home/ubuntu/workspace/data/terminals"
        session_dir = os.path.join(base_path, session_id)
        os.makedirs(session_dir, exist_ok=True)
        self.screen = pyte.Screen(self.width, self.height)
        self.stream = pyte.Stream(self.screen)
        
        # 2. 创建 PTY
        self.master, self.slave = pty.openpty()
        
        # 3. 启动子进程
        env = os.environ.copy()
        env["PROMPT_COMMAND"] = ""
        if "VSCODE_GIT_IPC_HANDLE" in env: del env["VSCODE_GIT_IPC_HANDLE"]
        env["TERM"] = "linux"  # 关键：让 TUI 软件识别
        
        self.proc = subprocess.Popen(
            [shell],
            stdin=self.slave,
            stdout=self.slave,
            stderr=self.slave,
            start_new_session=True,
            env=env,
            cwd=session_dir
        )

        # 4. 开启后台监控线程 (解决长时间运行、缓冲区排空问题)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _update_loop(self):
        """后台线程：实时同步终端输出到内存屏幕"""
        while not self._stop_event.is_set():
            # 使用 select 监控 master 描述符，超时设为 0.1s
            r, _, _ = select.select([self.master], [], [], 0.1)
            if r:
                try:
                    data = os.read(self.master, 4096)
                    if not data:
                        break
                    # 解析 ANSI 转义码
                    self.stream.feed(data.decode(errors='ignore'))
                except (OSError, EOFError):
                    break
            if self.proc.poll() is not None: # 进程已退出
                break

    def send(self, data: str):
        """发送数据到终端 (非阻塞)"""
        if self.proc.poll() is None:
            os.write(self.master, data.encode())

    def get_status(self) -> str:
        """获取当前屏幕快照 (LLM 友好格式)"""
        lines = []
        # 直接读取内存中 pyte 维护的屏幕状态
        for i, line in enumerate(self.screen.display):
            content = line.rstrip()
            # 标注光标
            if i == self.screen.cursor.y:
                x = self.screen.cursor.x
                content = f"{content[:x]}█{content[x:]} (CURSOR)"
            
            # 仅保留有内容的行，节省 LLM Token
            if content.strip() or i == self.screen.cursor.y:
                lines.append(f"Line {i:02}: {content}")
        
        return "\n".join(lines)

    def close(self):
        """释放资源并杀死进程"""
        self._stop_event.set()
        try:
            self.proc.terminate()
            os.close(self.master)
            os.close(self.slave)
        except:
            pass

# === 资源管理器 (Session Manager) ===

class SessionManager:
    """通用资源管理器：支持多终端、多类型的 Session"""
    def __init__(self):
        self._sessions: Dict[str, BaseSession] = {}
        self._lock = threading.Lock()

    def get_or_create_terminal(self, session_id: str, **kwargs) -> TerminalSession:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = TerminalSession(session_id, **kwargs)
            return self._sessions[session_id]

    def get_session(self, session_id: str) -> Optional[BaseSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> List[str]:
        with self._lock:
            return list(self._sessions.keys())

    def close_session(self, session_id: str):
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                session.close()

# 单例模式，供 FastAPI 全局调用
manager = SessionManager()