
# src/handlers/think_relay_handlers.py
import os
import json
import time
from fastapi import HTTPException

from src.utils.file_utils import write_json_file_atomic

# DOC-BEGIN id=think-relay/file-path#1 type=config v=1
# summary: Think relay 数据文件路径，存放在项目根 data/ 目录下
# intent: 统一放在 DATA_DIR 下，避免散落；与 relay_current_plan 平级
THINK_RELAY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "think_relay.json"
)
# DOC-END id=think-relay/file-path#1


# DOC-BEGIN id=think-relay/models#1 type=design v=1
# summary: ThinkRelaySetReq 接收 parts（act 未完成条目扁平列表）和 source 字符串；
#   ThinkRelayGetReq 无字段，仅作占位符与 remote_plan_get 保持一致风格
# intent: parts 字段名从 nodes 改为 parts，与前端 ThinkEventPart 概念对齐，
#   减少跨端理解歧义；后端仍只做透明存储
from pydantic import BaseModel, Field
from typing import Any

class ThinkRelaySetReq(BaseModel):
    parts: list[dict[str, Any]] = Field(default_factory=list)
    source: str = ""

class ThinkRelayGetReq(BaseModel):
    pass
# DOC-END id=think-relay/models#1


# DOC-BEGIN id=think-relay/handlers#1 type=logic v=1
# summary: handle_think_relay_set 接收 act 未完成 parts 列表写入文件；
#   handle_think_relay_get 读取文件原样返回，文件不存在时返回安全空结构。
#   写入格式：{ updated_at, source, parts: [{id, itemId, itemTitle, title, description, editableContent}, ...] }
#   读取格式：{ ok, exists, data: { updated_at, source, parts, results? } }
# intent: results 字段由另一端思考前端写入，每条结构为与原 part 相同的完整对象（含同一 id），
#   本端 pull 时原样返回，前端按 id 全量替换对应 part，后端不感知替换逻辑
async def handle_think_relay_set(data: ThinkRelaySetReq) -> dict:
    # DOC-BEGIN id=think-relay/set-build-payload#1 type=behavior v=1
    # summary: 构建 payload 只包含 updated_at、source、parts 三个字段
    # intent: results 概念已移除，思考内容直接写入 parts[].editableContent
    payload = {
        "updated_at": time.time(),
        "source": data.source or "",
        "parts": data.parts,
    }
    # DOC-END id=think-relay/set-build-payload#1
    write_json_file_atomic(THINK_RELAY_FILE, payload)
    return {"ok": True, "status": "stored", "updated_at": payload["updated_at"], "count": len(data.parts)}


async def handle_think_relay_get(_: ThinkRelayGetReq) -> dict:
    if not os.path.exists(THINK_RELAY_FILE):
        return {
            "ok": True,
            "exists": False,
            "data": {"updated_at": None, "source": "", "parts": []}
        }
    try:
        with open(THINK_RELAY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"ok": True, "exists": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read think relay file: {e}")


# DOC-END id=think-relay/handlers#1
