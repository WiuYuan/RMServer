# src/services/blog_job_queue.py
# ============================================================
# Global FIFO job queue for blog generation
# ============================================================

import logging
import threading
import queue
from concurrent.futures import Future
from typing import Callable, Any, Tuple

logger = logging.getLogger(__name__)


class BlogJobQueue:
    """
    A global FIFO job queue with single worker.
    All blog generation tasks are executed sequentially.
    """

    def __init__(self):
        self._queue: queue.Queue[
            Tuple[Callable[[], Any], Future]
        ] = queue.Queue()

        self._worker = threading.Thread(
            target=self._run,
            daemon=True,
        )
        self._worker.start()

    # DOC-BEGIN id=blog-queue/run#1 type=function v=1
    # summary: 消费循环从队列中逐个取出 (func, future) 对，执行 func 并将结果/异常设置到 future 上；
    #   如果 future 已被 cancel（如 clear() 后残留任务），set_result/set_exception 会抛 InvalidStateError，
    #   外层 except Exception 捕获后忽略，确保消费线程永远不会死亡
    # intent: 之前版本中如果 future.set_result() 抛异常（cancelled future），整个 _run 线程会崩溃，
    #   导致后续所有提交的任务永远不会被执行——这就是 cancel_all 之后再提交任务"没有回音"的根因。
    #   现在用双层 try-except 保证线程存活。
    # DOC-BEGIN id=blog-queue/run-loop#1 type=behavior v=2
    # summary: 消费线程主循环：从队列取任务 → 检查是否已取消 → 执行 → 设置结果。每个关键节点都输出日志。
    # intent: 之前 worker 线程可能静默死亡或任务静默丢弃，无法排查；现在每个阶段都有日志，
    #   便于在终端/日志文件中追踪任务生命周期。双层 try-except 保证线程永远不会死亡。
    def _run(self):
        logger.info("[BlogJobQueue] Worker thread started, waiting for jobs...")
        while True:
            logger.info(f"[BlogJobQueue] Waiting for next job... (queue size={self._queue.qsize()})")
            func, future = self._queue.get()
            if future.cancelled():
                logger.warning("[BlogJobQueue] Job was cancelled before execution, skipping.")
                self._queue.task_done()
                continue
            logger.info(f"[BlogJobQueue] Picked up job: {func}")
            try:
                result = func()
                logger.info("[BlogJobQueue] Job completed successfully.")
                try:
                    future.set_result(result)
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"[BlogJobQueue] Job raised exception: {e}", exc_info=True)
                try:
                    future.set_exception(e)
                except Exception:
                    pass
            finally:
                self._queue.task_done()
    # DOC-END id=blog-queue/run-loop#1

    # DOC-BEGIN id=blog-queue/clear#1 type=behavior v=1
    # summary: 清空队列中所有尚未开始执行的任务，将它们的 Future 设为 CancelledError；
    #   正在执行的任务不受影响，会继续跑完
    # intent: Cancel All 功能需要清空待执行队列防止新任务继续启动；
    #   queue.Queue 没有原生 clear 方法，通过循环 get_nowait 逐个取出并取消；
    #   已经在 _run 中取出的任务无法被 clear 触及，这是预期行为
    def clear(self):
        cancelled = 0
        while not self._queue.empty():
            try:
                func, future = self._queue.get_nowait()
                future.cancel()
                self._queue.task_done()
                cancelled += 1
            except Exception:
                break
        return cancelled
    # DOC-END id=blog-queue/clear#1

    def submit(self, func: Callable[[], Any]) -> Future:
        """
        Submit a job. The job will be executed after all previous jobs finish.
        """
        future = Future()
        self._queue.put((func, future))
        return future


# ============================================================
# Global singleton
# ============================================================

BLOG_JOB_QUEUE = BlogJobQueue()
