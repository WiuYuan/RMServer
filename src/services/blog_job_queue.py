# ============================================================
# Global FIFO job queue for blog generation
# ============================================================

import threading
import queue
from concurrent.futures import Future
from typing import Callable, Any, Tuple


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

    def _run(self):
        while True:
            func, future = self._queue.get()
            try:
                result = func()
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                self._queue.task_done()

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
