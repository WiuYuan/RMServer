# src/services/task_session.py
from pathlib import Path
import json
from src.services.agents import Tool_Calls

class TaskSession:
    def __init__(self, base_dir: str, task_id: str):
        self.task_id = task_id
        self.dir = Path(base_dir) / task_id
        self.dir.mkdir(parents=True, exist_ok=True)

        self.messages_path = self.dir / "messages.json"
        self.meta_path = self.dir / "meta.json"

        self.tool_calls = Tool_Calls(
            LOG_DIR=str(self.dir),
            MAX_CHAR=20000,
            mode="Summary",
        )

        if not self.messages_path.exists():
            self._init_task()

    def _init_task(self):
        with open(self.meta_path, "w") as f:
            json.dump({"task_id": self.task_id}, f)

        with open(self.messages_path, "w") as f:
            json.dump([], f)

    def load_messages(self):
        return json.load(open(self.messages_path))

    def save_messages(self, messages):
        json.dump(messages, open(self.messages_path, "w"), ensure_ascii=False, indent=2)