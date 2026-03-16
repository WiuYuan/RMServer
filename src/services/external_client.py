# src/services/external_client.py
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

    # DOC-BEGIN id=external_client/send_image#1 type=behavior v=1
    # summary: 向队列推送一个 type="image" 的事件，包含图片的 base64 数据、MIME 类型和展示名称
    # intent: § Exec push 指令拦截后，由 workspace_edit_parser 调用此方法将图片数据推入队列。
    #   前端 event_generator 遇到 type="image" 时需要特殊处理（不作为普通文本 yield），
    #   而是序列化为 JSON 让前端识别并渲染为 <img>。
    #   mime_type 由调用方根据文件扩展名推断，默认 image/png。
    def send_image(self, name: str, data_base64: str, mime_type: str = "image/png"):
        if self.out_queue is None:
            return
        self.out_queue.put({
            "type": "image",
            "data": {
                "name": name,
                "mime_type": mime_type,
                "base64": data_base64,
            }
        })
    # DOC-END id=external_client/send_image#1
