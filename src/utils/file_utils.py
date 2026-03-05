import os
import json
from typing import Any

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    
def try_acquire_lock(lock_path: str) -> bool:
    """
    原子创建 lock 文件
    成功：返回 True
    已存在：返回 False
    """
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False
    
def release_lock(lock_path: str):
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass

def read_text_file(p: str) -> str:
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def write_text_file(p: str, content: str):
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)

# DOC-BEGIN id=manual/relay/json-atomic-write#1 type=logic v=1
# summary: 将 JSON 数据以“原子写入”方式落盘，避免并发/中断导致文件损坏
# intent: 作为“远端中转器”，文件会被多个请求读写：
#   - 先写入 .tmp，再 os.replace 覆盖，保证读者要么读到旧完整文件，要么读到新完整文件
#   - 自动创建父目录，降低部署出错概率
#   - 不引入文件锁：这里目标是“最后写入者为准”的中转语义，冲突由调用方约束（例如前端只推送当前计划）
def write_json_file_atomic(p: str, obj: Any):
    ensure_dir(os.path.dirname(p))
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
# DOC-END id=manual/relay/json-atomic-write#1

def list_files_in_dir(root: str):
    out = []
    if not os.path.isdir(root): 
        return out
    
    try:
        # 按修改时间倒序（最新的在前）
        files = sorted(os.listdir(root), key=lambda x: os.path.getmtime(os.path.join(root, x)), reverse=True)
    except:
        files = os.listdir(root)

    for fname in files:
        # 只列出 html 文件
        if fname.lower().endswith(".html"):
            out.append({
                "article_id": fname,          # ID 就是文件名 (e.g. "random_name.html")
                "title": fname[:-5]           # Title 只是用来展示 (e.g. "random_name")
            })
    return out