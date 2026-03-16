# src/services/task_manager.py
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any

# DOC-BEGIN id=task_manager/task_ctrl#1 type=design v=2
# summary: TaskCtrl 持有 stop（外部请求停止信号）和 done（LLM 线程已退出信号）两个 Event；
#   ensure_task 创建或复用 TaskCtrl，stop_task 设置 stop 并等待 done 或超时
# intent: stop 和 done 分离：stop 是"请求停止"的单向信号（由 API 层设置），
#   done 是"线程已退出"的确认信号（由 _run_fn finally 设置）。
#   stop_task 返回前等待 done，确保调用方拿到响应时 LLM 线程已真正退出，
#   SSE 流也已发送 None（EOS），前端不会出现"停止了但仍在 loading"的状态。
#   超时 15 秒兜底：tool call 可能阻塞在 sleep/网络请求中，poll 间隔 50ms 检测 stop，
#   最坏情况需等 30 秒（wait_seconds 上限），15 秒覆盖绝大多数场景。
@dataclass
class TaskCtrl:
    stop: threading.Event
    done: threading.Event

TASKS: Dict[str, TaskCtrl] = {}
TASKS_LOCK = threading.Lock()

def ensure_task(task_id: str) -> TaskCtrl:
    with TASKS_LOCK:
        ctrl = TASKS.get(task_id)
        if not ctrl:
            ctrl = TaskCtrl(stop=threading.Event(), done=threading.Event())
            TASKS[task_id] = ctrl
        return ctrl

def stop_task(task_id: str, timeout: float = 15.0) -> bool:
    with TASKS_LOCK:
        ctrl = TASKS.get(task_id)
    if not ctrl:
        return True
    ctrl.stop.set()
    finished = ctrl.done.wait(timeout=timeout)
    return finished
# DOC-END id=task_manager/task_ctrl#1
    
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

def run_query_with_tools_safe(fn, result_queue):
    try: fn()
    except Exception: traceback.print_exc()
    finally: result_queue.put(None)
