# rmstop.sh
# DOC-BEGIN id=scripts/stop#1 type=ops v=1
# summary: 先通过 API 列出所有 terminal 并逐个关闭，再通过 PID 文件终止 uvicorn 进程
# intent: 直接 kill 进程会导致 PTY 子进程(bash)变成孤儿进程、文件描述符泄漏。
#   先调 terminal_close API 让 SessionManager 执行 close()（terminate子进程+关闭fd），
#   再优雅终止主进程。API_KEY 从 secret 文件读取，与 server.py 逻辑一致。
#   超时 5s 后 SIGKILL 强杀，兜底防止进程卡死。
# DOC-END id=scripts/stop#1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="server.pid"
SECRET_FILE="$HOME/.rmserver/.env_secret"
API_BASE="http://127.0.0.1:8000"

# 读取 API Key
if [ -f "$SECRET_FILE" ]; then
    API_KEY=$(cat "$SECRET_FILE" | tr -d '[:space:]')
else
    API_KEY="default_insecure_password"
fi

# --- 阶段1: 销毁所有 workspace ---
echo "[INFO] Destroying all workspaces..."

WORKSPACES_JSON=$(curl -s -X POST "$API_BASE/gateway" \
    -H "Content-Type: application/json" \
    -H "X-Server-Api-Key: $API_KEY" \
    -d '{"action": "workspace_list", "data": {}}' 2>/dev/null)

if [ $? -eq 0 ] && [ -n "$WORKSPACES_JSON" ]; then
    WS_IDS=$(echo "$WORKSPACES_JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for ws in data.get('workspaces', []):
        wid = ws.get('workspace_id', '') if isinstance(ws, dict) else ''
        if wid:
            print(wid)
except:
    pass
" 2>/dev/null)

    for WID in $WS_IDS; do
        echo "  [INFO] Deleting workspace: $WID"
        curl -s -X POST "$API_BASE/gateway" \
            -H "Content-Type: application/json" \
            -H "X-Server-Api-Key: $API_KEY" \
            -d "{\"action\": \"workspace_delete\", \"data\": {\"workspace_id\": \"$WID\"}}" > /dev/null 2>&1
    done
    echo "[OK] All workspaces destroyed"
else
    echo "[WARN] Cannot fetch workspace list, skipping workspace cleanup"
fi

# --- 阶段2: 销毁所有 terminal ---
echo "[INFO] Destroying all terminals..."

TERMINALS_JSON=$(curl -s -X POST "$API_BASE/gateway" \
    -H "Content-Type: application/json" \
    -H "X-Server-Api-Key: $API_KEY" \
    -d '{"action": "terminal_list_all", "data": {}}' 2>/dev/null)

if [ $? -eq 0 ] && [ -n "$TERMINALS_JSON" ]; then
    SESSION_IDS=$(echo "$TERMINALS_JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    terminals = data.get('terminals', [])
    for t in terminals:
        sid = t.get('session_id', '') if isinstance(t, dict) else t
        if sid:
            print(sid)
except:
    pass
" 2>/dev/null)

    for SID in $SESSION_IDS; do
        echo "  [INFO] Closing terminal: $SID"
        curl -s -X POST "$API_BASE/gateway" \
            -H "Content-Type: application/json" \
            -H "X-Server-Api-Key: $API_KEY" \
            -d "{\"action\": \"terminal_close\", \"data\": {\"session_id\": \"$SID\"}}" > /dev/null 2>&1
    done
    echo "[OK] All terminals destroyed"
else
    echo "[WARN] Cannot connect to server or fetch terminal list (server may not be running), skipping terminal cleanup"
fi

# --- 阶段3: 终止主进程 ---
if [ ! -f "$PID_FILE" ]; then
    echo "[WARN] $PID_FILE not found, trying to find process by port..."
    PID=$(lsof -ti:8000 2>/dev/null | head -1)
    if [ -z "$PID" ]; then
        echo "[INFO] No running server process found"
        exit 0
    fi
else
    PID=$(cat "$PID_FILE")
fi

if kill -0 "$PID" 2>/dev/null; then
    echo "[INFO] Terminating process (PID=$PID)..."
    kill "$PID"

    # 等待最多 5 秒优雅退出
    for i in $(seq 1 50); do
        if ! kill -0 "$PID" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done

    # 如果还活着，强杀
    if kill -0 "$PID" 2>/dev/null; then
        echo "[WARN] Process did not respond to SIGTERM, sending SIGKILL..."
        kill -9 "$PID"
    fi

    echo "[OK] Process terminated"
else
    echo "[INFO] Process (PID=$PID) is already gone"
fi

rm -f "$PID_FILE"
