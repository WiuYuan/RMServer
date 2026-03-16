# src/services/terminal.py
import pty
import os
import time
import pyte
import select
import subprocess
import threading
from typing import Dict, Optional, Type, List
from src.config import TERMINALS_ROOT

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

        # DOC-BEGIN id=terminal/runtime/pty_process_and_stop_event#1 type=core v=1
        # summary: 初始化终端运行时资源：创建 _stop_event；打开 PTY(master/slave)；启动 shell 子进程并绑定到 slave；设置 TERM/COLUMNS/LINES；最后启动后台读取线程将输出持续灌入 pyte 屏幕
        # intent: 修复 close() 中访问 _stop_event 报错的问题，并补齐 TerminalSession 所依赖的关键属性（proc/master/slave），避免 send()/close() 访问不存在字段；通过 PTY 让 shell 以交互方式运行，保证 prompt/控制字符输出符合预期；使用后台线程持续读取输出，避免阻塞主线程与丢失屏幕状态
        self._stop_event = threading.Event()

        self.master, self.slave = pty.openpty()

        env = os.environ.copy()
        env.setdefault("TERM", "xterm-256color")
        env["COLUMNS"] = str(self.width)
        env["LINES"] = str(self.height)

        self.proc = subprocess.Popen(
            [shell],
            stdin=self.slave,
            stdout=self.slave,
            stderr=self.slave,
            env=env,
            start_new_session=True,
        )
        # DOC-END id=terminal/runtime/pty_process_and_stop_event#1

        # 1. 初始化 Pyte 虚拟屏幕
        base_path = TERMINALS_ROOT
        session_dir = os.path.join(base_path, session_id)
        os.makedirs(session_dir, exist_ok=True)
        self.screen = pyte.Screen(self.width, self.height)
        self.stream = pyte.Stream(self.screen)

        # DOC-BEGIN id=terminal/command_recording#1 type=state v=2
        # summary: 命令录制状态——记录该 terminal 创建以来所有通过 send() 发送的命令及其时间间隔
        # intent: _command_log 是一个列表，每个元素 {"data": str, "elapsed": float} 表示
        #   自上一条命令（或 terminal 创建）以来经过的秒数以及发送的原始数据。
        #   _last_send_time 记录上一次 send 的时间戳，用于计算 elapsed。
        #   _recording_start 记录 terminal 创建时间，第一条命令的 elapsed 基于此计算。
        #   只记录用户主动发送的命令（record=True），不记录 marker 协议等内部操作（record=False）。
        self._command_log: List[dict] = []
        self._recording_start: float = time.time()
        self._last_send_time: float = self._recording_start
        self._record_lock = threading.Lock()
        # DOC-END id=terminal/command_recording#1

        # DOC-BEGIN id=terminal/runtime/start_update_thread#1 type=behavior v=1
        # summary: 启动后台更新线程 _update_loop，持续读取 PTY master 输出并 feed 到 pyte stream，从而让 get_status()/get_raw_screen() 始终反映最新屏幕状态
        # intent: 终端输出是异步的；如果不启动读取线程，屏幕不会更新，回放/提示符检测也会失效；daemon=True 避免服务退出时被线程阻塞，但也意味着必须依赖 close() 设置 stop_event 来尽快释放资源
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        # DOC-END id=terminal/runtime/start_update_thread#1

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

    # DOC-BEGIN id=terminal/send_with_recording#1 type=behavior v=1
    # summary: 发送数据到终端并记录到命令日志（支持 record 参数控制是否录制）
    # intent: record=True 时记录该命令和时间间隔到 _command_log，用于后续保存/回放。
    #   record=False 用于内部操作（如 marker 协议的 exec_command_reliable），避免污染录制。
    #   elapsed 计算基于上一次 send 的时间戳，第一条命令基于 terminal 创建时间。
    def send(self, data: str, record: bool = True):
        """发送数据到终端 (非阻塞)"""
        if self.proc.poll() is None:
            os.write(self.master, data.encode())
            if record:
                now = time.time()
                with self._record_lock:
                    elapsed = now - self._last_send_time
                    self._command_log.append({"data": data, "elapsed": round(elapsed, 3)})
                    self._last_send_time = now
    # DOC-END id=terminal/send_with_recording#1

    # DOC-BEGIN id=terminal/get_command_log#1 type=query v=1
    # summary: 获取该 terminal 创建以来的完整命令录制日志
    # intent: 返回 _command_log 的深拷贝，避免外部修改影响内部状态。
    def get_command_log(self) -> List[dict]:
        with self._record_lock:
            return [entry.copy() for entry in self._command_log]
    # DOC-END id=terminal/get_command_log#1

    # DOC-BEGIN id=terminal/clear_command_log#1 type=behavior v=1
    # summary: 清空命令录制日志并重置计时起点
    # intent: 用户保存录制后可能想清空重新开始。重置 _last_send_time 确保下一条命令的 elapsed 从清空时刻算起。
    def clear_command_log(self):
        with self._record_lock:
            self._command_log.clear()
            self._last_send_time = time.time()
    # DOC-END id=terminal/clear_command_log#1

    # DOC-BEGIN id=terminal/is_prompt_ready#1 type=query v=1
    # summary: 判断终端当前是否处于"等待用户输入"状态（即光标所在行看起来像一个 shell prompt）
    # intent: 回放命令时需要判断是否可以提前发送下一条命令（而不必等完整的间隔时间）。
    #   检测策略：光标所在行去除空白后以 $、#、>、% 结尾（常见 shell prompt 结尾字符），
    #   且该行后面没有更多非空行（排除命令输出中恰好有这些字符的情况）。
    #   这是启发式判断，不保证 100% 准确，但对 bash/zsh/sh 等主流 shell 足够可靠。
    def is_prompt_ready(self) -> bool:
        cursor_y = self.screen.cursor.y
        cursor_line = self.screen.display[cursor_y].rstrip()

        # 光标行必须非空且以 prompt 字符结尾
        stripped = cursor_line.rstrip()
        if not stripped:
            return False
        prompt_endings = ('$', '#', '>', '%', '❯')
        if not any(stripped.endswith(ch) for ch in prompt_endings):
            return False

        # 光标行之后不应有非空行（排除正在输出中的误判）
        for i in range(cursor_y + 1, min(cursor_y + 5, self.screen.lines)):
            if self.screen.display[i].strip():
                return False
        return True
    # DOC-END id=terminal/is_prompt_ready#1

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

    # DOC-BEGIN id=terminal/get_llm_status#1 type=query v=2
    # summary: 为 LLM 上下文裁剪终端屏幕快照——限制最大宽度(max_width)、仅保留尾部 tail_lines 行，
    #   最终返回格式与 get_status() 类似的带行号文本
    # intent: PTY 屏幕是 1000×1000，这是为 exec_command_reliable 等内部函数设计的大缓冲区。
    #   但 LLM 不需要看全量屏幕：(1) 宽度超过 120 字符的终端输出极少（标准终端 80 列），截断不丢失关键信息；
    #   (2) 高度上，LLM 只需看最近的输出（tail 80 行），头部 banner/motd 对决策无帮助，
    #   超出部分以 "... (N lines omitted) ..." 标记压缩。
    #   max_total_chars=8000（约 2000 token）作为最终安全阀，单个终端不会占用过多上下文。
    def get_llm_status(self, max_width: int = 120,
                       tail_lines: int = 80, max_total_chars: int = 8000) -> str:
        # 第一步：收集所有有内容的行（与 get_status 类似，但做宽度截断）
        content_lines = []
        for i, line in enumerate(self.screen.display):
            content = line.rstrip()
            is_cursor_line = (i == self.screen.cursor.y)

            # 标注光标
            if is_cursor_line:
                x = self.screen.cursor.x
                content = f"{content[:x]}█{content[x:]} (CURSOR)"

            # 仅保留有内容的行或光标行
            if content.strip() or is_cursor_line:
                # 宽度截断（光标标记后再截断，避免截掉光标信息）
                if len(content) > max_width:
                    content = content[:max_width] + "…"
                content_lines.append((i, content))

        if not content_lines:
            return "(empty screen)"

        # 第二步：高度裁剪——仅保留尾部 tail_lines 行
        total = len(content_lines)
        if total <= tail_lines:
            selected = content_lines
        else:
            omitted = total - tail_lines
            selected = [(-1, f"... ({omitted} lines omitted) ...")] + content_lines[-tail_lines:]

        # 第三步：格式化输出
        result_lines = []
        for line_no, content in selected:
            if line_no == -1:
                result_lines.append(content)
            else:
                result_lines.append(f"Line {line_no:02}: {content}")

        result = "\n".join(result_lines)

        # 第四步：总字符数安全阀
        if len(result) > max_total_chars:
            result = result[-max_total_chars:]
            # 从第一个完整行开始，避免截断行首
            first_newline = result.find("\n")
            if first_newline != -1 and first_newline < 200:
                result = "... (truncated) ...\n" + result[first_newline + 1:]
            else:
                result = "... (truncated) ...\n" + result

        return result
    # DOC-END id=terminal/get_llm_status#1

    def get_raw_screen(self) -> str:
        """返回屏幕纯文本（无行号、无光标标记），供程序解析用"""
        lines = []
        for line in self.screen.display:
            lines.append(line.rstrip())
        while lines and not lines[-1]:
            lines.pop()
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