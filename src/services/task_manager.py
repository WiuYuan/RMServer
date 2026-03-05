import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any

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

def run_query_with_tools_safe(fn, result_queue):
    try: fn()
    except Exception: traceback.print_exc()
    finally: result_queue.put(None)
