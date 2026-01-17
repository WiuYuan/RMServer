import queue
from typing import Optional, Dict, Any


class ExternalClient:
    """
    ExternalClient
    ----------------
    A lightweight event emitter that pushes structured messages
    into a thread-safe queue.

    It does NOT:
      - print
      - stream HTTP
      - know about FastAPI / UI

    It ONLY:
      - accept structured messages
      - put them into a queue
    """

    def __init__(self, out_queue: Optional[queue.Queue] = None):
        self.out_queue = out_queue

    def send_message(self, message: Dict[str, Any]):
        """
        Expected message format:
        {
            "type": "llm_info",
            "data": {
                "content": "..."
            }
        }
        """
        if self.out_queue is None:
            return
        
        if message is None:
            self.out_queue.put(None)

        # Basic validation (lightweight, non-strict)
        if not isinstance(message, dict):
            return
        if "type" not in message or "data" not in message:
            return

        self.out_queue.put(message)