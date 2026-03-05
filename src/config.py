import os

SECRET_FILE = "/home/ubuntu/.env_secret"
DATA_DIR = "/home/ubuntu/workspace/data"
ARTICLES_ROOT = f"{DATA_DIR}/articles"
PLAYBOOK_ROOT = f"{DATA_DIR}/playbook"
PLAYBOOK_ACTIONS_NEED_ROOT = {
    "playbook_list",
    "playbook_get",
    "playbook_upsert_policy",
    "playbook_upsert_backlog",
    "playbook_upsert_assumption",
    "playbook_delete",
    "playbook_suggest_next_step",
}

# DOC-BEGIN id=manual/relay/plan-storage-paths#1 type=design v=1
# summary: 定义“远端中转器”计划数据的落盘路径（DATA_DIR 下）
# intent: 你需要远端作为中转器供另一个前端拉取，因此必须落盘到稳定文件而非内存；
#   放在 DATA_DIR/relay 下便于与 articles/playbook/tasks 等数据分区隔离；
#   文件路径集中定义，便于未来扩展（例如加 settings、加多用户命名空间）。
RELAY_DIR = os.path.join(DATA_DIR, "relay")
RELAY_CURRENT_PLAN_FILE = os.path.join(RELAY_DIR, "current_plan.json")
# DOC-END id=manual/relay/plan-storage-paths#1

def with_default_playbook_root(action: str, data: dict) -> dict:
    """
    Keep request schemas unchanged, but allow clients to omit playbook_root.
    If missing/empty, inject server default PLAYBOOK_ROOT.
    """
    if not isinstance(data, dict):
        return data
    if action in PLAYBOOK_ACTIONS_NEED_ROOT:
        if not data.get("playbook_root"):
            data = {**data, "playbook_root": PLAYBOOK_ROOT}
    return data