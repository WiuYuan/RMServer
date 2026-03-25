# src/services/tts_job_queue.py
# ============================================================
# Global FIFO job queue for TTS generation
# ============================================================

import logging
import threading
import queue
from concurrent.futures import Future
from typing import Callable, Any, Tuple

logger = logging.getLogger(__name__)


# DOC-BEGIN id=tts-queue/class#1 type=design v=1
# summary: TTS专用的全局FIFO单worker队列，结构与BlogJobQueue完全一致，但独立运行，
#   使得TTS任务和Blog生成任务可以并行执行（各自队列内部串行）
# intent: TTS生成涉及LLM调用+Fish Audio API调用，耗时较长；与blog生成共享队列会导致
#   两类任务互相阻塞；独立队列让用户可以同时生成blog和TTS而不互相等待
class TTSJobQueue:
    """
    A global FIFO job queue with single worker for TTS tasks.
    All TTS generation tasks are executed sequentially.
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

    def _run(self):
        logger.info("[TTSJobQueue] Worker thread started, waiting for jobs...")
        while True:
            logger.info(f"[TTSJobQueue] Waiting for next job... (queue size={self._queue.qsize()})")
            func, future = self._queue.get()
            if future.cancelled():
                logger.warning("[TTSJobQueue] Job was cancelled before execution, skipping.")
                self._queue.task_done()
                continue
            logger.info(f"[TTSJobQueue] Picked up job: {func}")
            try:
                result = func()
                logger.info("[TTSJobQueue] Job completed successfully.")
                try:
                    future.set_result(result)
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"[TTSJobQueue] Job raised exception: {e}", exc_info=True)
                try:
                    future.set_exception(e)
                except Exception:
                    pass
            finally:
                self._queue.task_done()

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

    def submit(self, func: Callable[[], Any]) -> Future:
        future = Future()
        self._queue.put((func, future))
        return future
# DOC-END id=tts-queue/class#1


# ============================================================
# Global singleton
# ============================================================

TTS_JOB_QUEUE = TTSJobQueue()