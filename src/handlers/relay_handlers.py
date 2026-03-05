import os
import json
import time
from fastapi import HTTPException

from src.config import RELAY_CURRENT_PLAN_FILE
from src.utils.file_utils import write_json_file_atomic
from src.models.requests import RemotePlanSetReq, RemotePlanGetReq

# DOC-BEGIN id=manual/relay/remote-plan-handlers#1 type=logic v=1
# summary: 远端计划中转器：接收当前计划并落盘、读取并返回
# intent: 远端只做“中转存取”：
#   - set：写入带 updated_at/source 的包裹结构，便于另一端判断是否有更新
#   - get：文件不存在时返回空对象而非报错，便于新环境首次启动
async def handle_remote_plan_set(data: RemotePlanSetReq) -> dict:
    payload = {
        "updated_at": time.time(),
        "source": data.source or "",
        "plan": data.plan,
    }
    write_json_file_atomic(RELAY_CURRENT_PLAN_FILE, payload)
    return {"ok": True, "status": "stored", "updated_at": payload["updated_at"]}

async def handle_remote_plan_get(_: RemotePlanGetReq) -> dict:
    if not os.path.exists(RELAY_CURRENT_PLAN_FILE):
        return {"ok": True, "exists": False, "data": {"updated_at": None, "source": "", "plan": []}}

    try:
        with open(RELAY_CURRENT_PLAN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"ok": True, "exists": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read relay plan file: {e}")
# DOC-END id=manual/relay/remote-plan-handlers#1
 